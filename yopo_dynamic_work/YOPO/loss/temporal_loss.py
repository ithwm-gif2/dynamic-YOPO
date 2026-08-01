import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import cfg


class TimeIndexedBoxSafetyLoss(nn.Module):
    """Collision cost against ground and AABBs at each future timestamp."""

    def __init__(self, polynomial_map: torch.Tensor):
        super().__init__()
        self.register_buffer("polynomial_map", polynomial_map)
        self.sgm_time = float(cfg["sgm_time"])
        self.eval_points = int(cfg["temporal_eval_points"])
        self.temporal_peak_weight = float(cfg["temporal_peak_weight"])
        self.traj_num = int(cfg["traj_num"])
        self.d0 = float(cfg["d0"])
        self.r = float(cfg["r"])
        self.drone_radius = float(cfg["drone_radius"])
        self.map_half_x = float(cfg["temporal_map_half_extent_xy"][0])
        self.map_half_y = float(cfg["temporal_map_half_extent_xy"][1])
        self.map_ceiling = float(cfg["temporal_map_ceiling"])

    def trajectory_positions(
        self, fixed_derivatives: torch.Tensor, decision_derivatives: torch.Tensor
    ) -> torch.Tensor:
        derivatives = torch.cat(
            (fixed_derivatives, decision_derivatives), dim=2
        )
        coefficients = torch.matmul(
            derivatives, self.polynomial_map.transpose(0, 1)
        )
        times = torch.linspace(
            self.sgm_time / self.eval_points,
            self.sgm_time,
            self.eval_points,
            device=coefficients.device,
            dtype=coefficients.dtype,
        )
        powers = torch.stack(
            (
                torch.ones_like(times),
                times,
                times.square(),
                times.pow(3),
                times.pow(4),
                times.pow(5),
            ),
            dim=1,
        )
        return torch.einsum("baf,tf->bta", coefficients, powers)

    @staticmethod
    def signed_distance_to_aabb(
        points: torch.Tensor, centers: torch.Tensor, half_sizes: torch.Tensor
    ) -> torch.Tensor:
        q = (points - centers).abs() - half_sizes
        outside = torch.linalg.vector_norm(torch.clamp_min(q, 0.0), dim=-1)
        inside = torch.clamp_max(q.amax(dim=-1), 0.0)
        return outside + inside

    def forward(
        self,
        fixed_derivatives: torch.Tensor,
        decision_derivatives: torch.Tensor,
        future_boxes: torch.Tensor,
    ) -> torch.Tensor:
        flat_positions = self.trajectory_positions(
            fixed_derivatives, decision_derivatives
        )
        batch_size = future_boxes.shape[0]
        if flat_positions.shape[0] != batch_size * self.traj_num:
            raise ValueError("Trajectory and future-box batch dimensions disagree")

        positions = flat_positions.reshape(
            batch_size, self.traj_num, self.eval_points, 3
        )
        centers = future_boxes[..., 0:3]
        half_sizes = future_boxes[..., 3:6]
        box_distance = self.signed_distance_to_aabb(
            positions[:, :, :, None, :],
            centers[:, None, :, :, :],
            half_sizes[:, None, :, :, :],
        )
        nearest_box_distance = box_distance.amin(dim=-1) - self.drone_radius
        ground_distance = positions[..., 2] - self.drone_radius
        ceiling_distance = (
            self.map_ceiling - positions[..., 2] - self.drone_radius
        )
        x_boundary_distance = (
            self.map_half_x - positions[..., 0].abs() - self.drone_radius
        )
        y_boundary_distance = (
            self.map_half_y - positions[..., 1].abs() - self.drone_radius
        )
        nearest_distance = torch.stack(
            (
                nearest_box_distance,
                ground_distance,
                ceiling_distance,
                x_boundary_distance,
                y_boundary_distance,
            ),
            dim=-1,
        ).amin(dim=-1)

        exponent = -(nearest_distance - self.d0) / self.r
        collision_cost = torch.exp(torch.clamp(exponent, min=-20.0, max=20.0))
        mean_cost = collision_cost.mean(dim=-1)
        peak_cost = collision_cost.amax(dim=-1)
        combined_cost = mean_cost + self.temporal_peak_weight * peak_cost
        minimum_clearance = nearest_distance.amin(dim=-1).reshape(-1)
        return combined_cost.reshape(-1), minimum_clearance


class TemporalYOPOLoss(nn.Module):
    def __init__(self):
        super().__init__()
        polynomial_map, jerk_hessian, acceleration_hessian = self._qp_matrices()
        self.register_buffer("jerk_hessian", jerk_hessian)
        self.register_buffer("acceleration_hessian", acceleration_hessian)
        self.safety_loss = TimeIndexedBoxSafetyLoss(polynomial_map)
        self.goal_length = float(cfg["goal_length"])

        velocity_scale = float(cfg["vel_max_train"])
        self.smoothness_weight = float(cfg["ws"]) / velocity_scale ** 5
        self.acceleration_weight = float(cfg["wa"]) / velocity_scale ** 3
        self.safety_weight = float(cfg["wc"])
        self.goal_weight = float(cfg["wg"])

        print("------ Temporal Loss ------")
        print(f"| {'smooth':<12} = {self.smoothness_weight:6.4f} |")
        print(f"| {'safety':<12} = {self.safety_weight:6.4f} |")
        print(f"| {'goal':<12} = {self.goal_weight:6.4f} |")
        print("---------------------------")

    @staticmethod
    def _qp_matrices():
        segment_time = float(cfg["sgm_time"])
        boundary_map = torch.zeros((6, 6), dtype=torch.float32)
        for derivative in range(3):
            boundary_map[2 * derivative, derivative] = math.factorial(derivative)
            for power in range(derivative, 6):
                boundary_map[2 * derivative + 1, power] = (
                    math.factorial(power)
                    / math.factorial(power - derivative)
                    * segment_time ** (power - derivative)
                )

        jerk = torch.zeros((6, 6), dtype=torch.float32)
        acceleration = torch.zeros((6, 6), dtype=torch.float32)
        for row in range(3, 6):
            for column in range(3, 6):
                jerk[row, column] = (
                    row
                    * (row - 1)
                    * (row - 2)
                    * column
                    * (column - 1)
                    * (column - 2)
                    / (row + column - 5)
                    * segment_time ** (row + column - 5)
                )
        for row in range(2, 6):
            for column in range(2, 6):
                acceleration[row, column] = (
                    row
                    * (row - 1)
                    * column
                    * (column - 1)
                    / (row + column - 3)
                    * segment_time ** (row + column - 3)
                )

        permutation = torch.zeros((6, 6), dtype=torch.float32)
        permutation[[0, 2, 4, 1, 3, 5], [0, 1, 2, 3, 4, 5]] = 1.0
        inverse_boundary = torch.inverse(boundary_map)
        polynomial_map = inverse_boundary @ permutation
        jerk_hessian = (
            permutation.transpose(0, 1)
            @ inverse_boundary.transpose(0, 1)
            @ jerk
            @ inverse_boundary
            @ permutation
        )
        acceleration_hessian = (
            permutation.transpose(0, 1)
            @ inverse_boundary.transpose(0, 1)
            @ acceleration
            @ inverse_boundary
            @ permutation
        )
        return polynomial_map, jerk_hessian, acceleration_hessian

    def smoothness_cost(
        self, fixed_derivatives: torch.Tensor, decision_derivatives: torch.Tensor
    ):
        derivatives = torch.cat(
            (fixed_derivatives, decision_derivatives), dim=2
        )
        jerk = torch.einsum(
            "bai,ij,baj->b", derivatives, self.jerk_hessian, derivatives
        )
        acceleration = torch.einsum(
            "bai,ij,baj->b",
            derivatives,
            self.acceleration_hessian,
            derivatives,
        )
        return jerk, acceleration

    @staticmethod
    def guidance_cost(
        fixed_derivatives: torch.Tensor,
        decision_derivatives: torch.Tensor,
        goal: torch.Tensor,
    ) -> torch.Tensor:
        current_position = fixed_derivatives[:, :, 0]
        end_position = decision_derivatives[:, :, 0]
        trajectory_direction = end_position - current_position
        goal_direction = goal - current_position
        goal_norm = torch.linalg.vector_norm(goal_direction, dim=1, keepdim=True)
        goal_unit = goal_direction / goal_norm.clamp_min(1e-8)

        along = (trajectory_direction * goal_unit).sum(dim=1)
        parallel_difference = F.smooth_l1_loss(
            along, goal_norm.squeeze(1), reduction="none"
        )
        perpendicular = trajectory_direction - along[:, None] * goal_unit
        return parallel_difference + 0.5 * torch.linalg.vector_norm(
            perpendicular, dim=1
        )

    def forward(
        self,
        state: torch.Tensor,
        prediction: torch.Tensor,
        goal: torch.Tensor,
        future_boxes: torch.Tensor,
    ):
        fixed_derivatives = state.permute(0, 2, 1)
        decision_derivatives = prediction.permute(0, 2, 1)
        smoothness, acceleration = self.smoothness_cost(
            fixed_derivatives, decision_derivatives
        )
        safety, minimum_clearance = self.safety_loss(
            fixed_derivatives, decision_derivatives, future_boxes
        )
        guidance = self.guidance_cost(
            fixed_derivatives, decision_derivatives, goal
        )
        return (
            self.smoothness_weight * smoothness,
            self.safety_weight * safety,
            self.goal_weight * guidance,
            self.acceleration_weight * acceleration,
            minimum_clearance,
        )
