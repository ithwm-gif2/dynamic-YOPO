#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "sensor_simulator.cuh"

namespace fs = std::filesystem;
using raycast::CameraParams;
using raycast::DynamicBox;
using raycast::DynamicDepthRenderer;

namespace
{
constexpr float kPi = 3.14159265358979323846f;

struct MovingObstacle
{
    int id{0};
    bool is_dynamic{true};
    Eigen::Vector2f position{Eigen::Vector2f::Zero()};
    Eigen::Vector2f velocity{Eigen::Vector2f::Zero()};
    float size_x{1.0f};
    float size_y{1.0f};
    float height{4.0f};

    DynamicBox box() const
    {
        DynamicBox result;
        result.center_x = position.x();
        result.center_y = position.y();
        result.center_z = 0.5f * height;
        result.half_x = 0.5f * size_x;
        result.half_y = 0.5f * size_y;
        result.half_z = 0.5f * height;
        return result;
    }
};

struct DroneState
{
    Eigen::Vector3f position{Eigen::Vector3f::Zero()};
    Eigen::Vector3f velocity{Eigen::Vector3f::Zero()};
    Eigen::Vector3f acceleration{Eigen::Vector3f::Zero()};
    Eigen::Vector3f goal{Eigen::Vector3f::Zero()};
    float target_speed{4.0f};
};

struct GeneratorConfig
{
    fs::path save_path;
    int episode_num{10};
    int frames_per_episode{10000};
    int seed{3};
    float frame_rate{33.0f};
    int history_length{4};
    float future_horizon{1.7f};
    std::string scenario{"dynamic"};
    float dynamic_ratio{1.0f};

    float map_x{60.0f};
    float map_y{60.0f};
    float map_z{15.0f};
    float area_per_obstacle{30.0f};
    float obstacle_width_min{0.6f};
    float obstacle_width_max{1.5f};
    float obstacle_height_min{3.0f};
    float obstacle_height_max{6.0f};
    float obstacle_speed_min{0.5f};
    float obstacle_speed_max{3.0f};
    float obstacle_spawn_clearance{0.25f};
    float obstacle_drone_reflection_margin{0.05f};
    bool reflect_obstacles_at_drone{false};

    float drone_speed_min{2.0f};
    float drone_speed_max{6.0f};
    float drone_z_min{1.2f};
    float drone_z_max{4.0f};
    float drone_radius{0.3f};
    float drone_safe_distance{0.6f};
    float waypoint_tolerance{2.0f};
    float avoidance_radius{5.0f};
    float avoidance_gain{14.0f};
    float max_acceleration{6.0f};
    float velocity_time_constant{0.8f};
    float boundary_margin{4.0f};

    CameraParams camera;
    float camera_pitch_rad{0.0f};
};

float uniform(std::mt19937 &rng, float lower, float upper)
{
    std::uniform_real_distribution<float> distribution(lower, upper);
    return distribution(rng);
}

float clampNorm(Eigen::Vector3f &value, float max_norm)
{
    const float norm = value.norm();
    if (norm > max_norm && norm > 1e-6f)
        value *= max_norm / norm;
    return norm;
}

fs::path projectRoot()
{
    fs::path path(CONFIG_FILE_PATH);
    for (int i = 0; i < 4; ++i)
        path = path.parent_path();
    return path;
}

GeneratorConfig loadConfig()
{
    const YAML::Node root = YAML::LoadFile(CONFIG_FILE_PATH);
    const YAML::Node dynamic = root["dynamic_dataset"];
    if (!dynamic)
        throw std::runtime_error("Missing dynamic_dataset section in config.yaml");

    GeneratorConfig cfg;
    cfg.save_path = projectRoot() / dynamic["save_path"].as<std::string>();
    cfg.episode_num = dynamic["episode_num"].as<int>();
    cfg.frames_per_episode = dynamic["frames_per_episode"].as<int>();
    cfg.seed = dynamic["seed"].as<int>();
    cfg.frame_rate = dynamic["frame_rate"].as<float>();
    cfg.history_length = dynamic["history_length"].as<int>();
    cfg.future_horizon = dynamic["future_horizon"].as<float>();

    cfg.map_x = root["x_length"].as<float>();
    cfg.map_y = root["y_length"].as<float>();
    cfg.map_z = root["z_length"].as<float>();
    cfg.area_per_obstacle = dynamic["area_per_obstacle"].as<float>();
    cfg.obstacle_width_min = dynamic["obstacle_width_min"].as<float>();
    cfg.obstacle_width_max = dynamic["obstacle_width_max"].as<float>();
    cfg.obstacle_height_min = dynamic["obstacle_height_min"].as<float>();
    cfg.obstacle_height_max = dynamic["obstacle_height_max"].as<float>();
    cfg.obstacle_speed_min = dynamic["obstacle_speed_min"].as<float>();
    cfg.obstacle_speed_max = dynamic["obstacle_speed_max"].as<float>();
    cfg.obstacle_spawn_clearance = dynamic["obstacle_spawn_clearance"].as<float>();
    cfg.obstacle_drone_reflection_margin =
        dynamic["obstacle_drone_reflection_margin"].as<float>();
    cfg.reflect_obstacles_at_drone =
        dynamic["reflect_obstacles_at_drone"].as<bool>();

    cfg.drone_speed_min = dynamic["drone_speed_min"].as<float>();
    cfg.drone_speed_max = dynamic["drone_speed_max"].as<float>();
    cfg.drone_z_min = dynamic["drone_z_range"][0].as<float>();
    cfg.drone_z_max = dynamic["drone_z_range"][1].as<float>();
    cfg.drone_radius = dynamic["drone_radius"].as<float>();
    cfg.drone_safe_distance = dynamic["drone_safe_distance"].as<float>();
    cfg.waypoint_tolerance = dynamic["waypoint_tolerance"].as<float>();
    cfg.avoidance_radius = dynamic["avoidance_radius"].as<float>();
    cfg.avoidance_gain = dynamic["avoidance_gain"].as<float>();
    cfg.max_acceleration = dynamic["max_acceleration"].as<float>();
    cfg.velocity_time_constant = dynamic["velocity_time_constant"].as<float>();
    cfg.boundary_margin = dynamic["boundary_margin"].as<float>();

    cfg.camera.fx = root["camera"]["fx"].as<float>();
    cfg.camera.fy = root["camera"]["fy"].as<float>();
    cfg.camera.cx = root["camera"]["cx"].as<float>();
    cfg.camera.cy = root["camera"]["cy"].as<float>();
    cfg.camera.image_width = root["camera"]["image_width"].as<int>();
    cfg.camera.image_height = root["camera"]["image_height"].as<int>();
    cfg.camera.max_depth_dist = root["camera"]["max_depth_dist"].as<float>();
    cfg.camera.normalize_depth = false;
    cfg.camera_pitch_rad = root["camera"]["pitch"].as<float>() * kPi / 180.0f;
    return cfg;
}

void applyArguments(int argc, char **argv, GeneratorConfig &cfg)
{
    for (int i = 1; i < argc; ++i)
    {
        const std::string argument(argv[i]);
        auto requireValue = [&](const std::string &name) -> std::string
        {
            if (i + 1 >= argc)
                throw std::runtime_error("Missing value for " + name);
            return argv[++i];
        };

        if (argument == "--episodes")
            cfg.episode_num = std::stoi(requireValue(argument));
        else if (argument == "--frames")
            cfg.frames_per_episode = std::stoi(requireValue(argument));
        else if (argument == "--seed")
            cfg.seed = std::stoi(requireValue(argument));
        else if (argument == "--output")
        {
            cfg.save_path = fs::path(requireValue(argument));
            if (cfg.save_path.is_relative())
                cfg.save_path = projectRoot() / cfg.save_path;
        }
        else if (argument == "--scenario")
        {
            cfg.scenario = requireValue(argument);
            if (cfg.scenario == "dynamic")
                cfg.dynamic_ratio = 1.0f;
            else if (cfg.scenario == "mixed")
                cfg.dynamic_ratio = 0.5f;
            else if (cfg.scenario == "static")
                cfg.dynamic_ratio = 0.0f;
            else
                throw std::runtime_error(
                    "scenario must be dynamic, mixed, or static");
        }
        else if (argument == "--dynamic-ratio")
            cfg.dynamic_ratio = std::stof(requireValue(argument));
        else if (argument == "--help")
        {
            std::cout << "dynamic_dataset_generator [--episodes N] [--frames N] "
                      << "[--seed N] [--output PATH] "
                      << "[--scenario dynamic|mixed|static] [--dynamic-ratio R]"
                      << std::endl;
            std::exit(0);
        }
        else
            throw std::runtime_error("Unknown argument: " + argument);
    }
}

void validateConfig(const GeneratorConfig &cfg)
{
    if (cfg.episode_num <= 0 || cfg.frames_per_episode <= 0)
        throw std::runtime_error("episode_num and frames_per_episode must be positive");
    if (cfg.frame_rate <= 0.0f || cfg.history_length < 2 || cfg.future_horizon <= 0.0f)
        throw std::runtime_error("Invalid temporal dataset settings");
    if (cfg.area_per_obstacle <= 0.0f)
        throw std::runtime_error("area_per_obstacle must be positive");
    if (cfg.dynamic_ratio < 0.0f || cfg.dynamic_ratio > 1.0f)
        throw std::runtime_error("dynamic_ratio must be within [0, 1]");
    if (cfg.obstacle_speed_min < 0.5f || cfg.obstacle_speed_max > 3.0f ||
        cfg.obstacle_speed_min > cfg.obstacle_speed_max)
        throw std::runtime_error("Obstacle speed must remain within the requested [0.5, 3.0] m/s range");
    if (cfg.obstacle_width_min <= 0.0f || cfg.obstacle_width_min > cfg.obstacle_width_max)
        throw std::runtime_error("Invalid obstacle width range");
    if (cfg.drone_z_min <= 0.0f || cfg.drone_z_min > cfg.drone_z_max)
        throw std::runtime_error("Invalid drone z range");
}

void prepareSavePath(const fs::path &path)
{
    if (fs::exists(path))
    {
        std::cout << "Removing existing dynamic dataset: " << path << std::endl;
        fs::remove_all(path);
    }
    fs::create_directories(path);
}

void saveDepth16(const cv::Mat &depth, float max_depth, const fs::path &path)
{
    cv::Mat normalized;
    cv::min(depth / max_depth, 1.0, normalized);
    cv::max(normalized, 0.0, normalized);
    cv::Mat output;
    normalized.convertTo(output, CV_16UC1, 65535.0);
    if (!cv::imwrite(path.string(), output))
        throw std::runtime_error("Failed to save depth image: " + path.string());
}

void saveDynamicMask(const cv::Mat &depth,
                     const cv::Mat &static_depth,
                     const fs::path &path)
{
    cv::Mat mask;
    cv::compare(depth + 1e-3f, static_depth, mask, cv::CMP_LT);
    if (!cv::imwrite(path.string(), mask))
        throw std::runtime_error("Failed to save dynamic mask: " + path.string());
}

float signedDistanceToBox(const Eigen::Vector3f &point,
                          const MovingObstacle &obstacle,
                          float drone_radius)
{
    const Eigen::Vector3f center(obstacle.position.x(), obstacle.position.y(), 0.5f * obstacle.height);
    const Eigen::Vector3f half(0.5f * obstacle.size_x,
                               0.5f * obstacle.size_y,
                               0.5f * obstacle.height);
    const Eigen::Vector3f q = (point - center).cwiseAbs() - half;
    const Eigen::Vector3f outside = q.cwiseMax(0.0f);
    const float inside = std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f);
    return outside.norm() + inside - drone_radius;
}

float minimumClearance(const Eigen::Vector3f &position,
                       const std::vector<MovingObstacle> &obstacles,
                       float drone_radius)
{
    float minimum = std::numeric_limits<float>::max();
    for (const auto &obstacle : obstacles)
        minimum = std::min(minimum, signedDistanceToBox(position, obstacle, drone_radius));
    return minimum;
}

bool boxesOverlap(const MovingObstacle &a,
                  const MovingObstacle &b,
                  float clearance)
{
    return std::abs(a.position.x() - b.position.x()) <
               0.5f * (a.size_x + b.size_x) + clearance &&
           std::abs(a.position.y() - b.position.y()) <
               0.5f * (a.size_y + b.size_y) + clearance;
}

std::vector<MovingObstacle> generateObstacles(const GeneratorConfig &cfg,
                                              std::mt19937 &rng)
{
    const int obstacle_count = std::max(
        1, static_cast<int>(std::lround(cfg.map_x * cfg.map_y / cfg.area_per_obstacle)));
    std::vector<MovingObstacle> obstacles;
    obstacles.reserve(obstacle_count);
    const int dynamic_count = static_cast<int>(
        std::lround(cfg.dynamic_ratio * obstacle_count));
    std::vector<bool> dynamic_flags(obstacle_count, false);
    std::fill(dynamic_flags.begin(), dynamic_flags.begin() + dynamic_count, true);
    std::shuffle(dynamic_flags.begin(), dynamic_flags.end(), rng);

    for (int id = 0; id < obstacle_count; ++id)
    {
        bool accepted = false;
        MovingObstacle obstacle;
        obstacle.id = id;
        obstacle.is_dynamic = dynamic_flags[id];
        for (int attempt = 0; attempt < 20000 && !accepted; ++attempt)
        {
            obstacle.size_x = uniform(rng, cfg.obstacle_width_min, cfg.obstacle_width_max);
            obstacle.size_y = uniform(rng, cfg.obstacle_width_min, cfg.obstacle_width_max);
            obstacle.height = uniform(rng, cfg.obstacle_height_min, cfg.obstacle_height_max);
            const float x_limit = 0.5f * cfg.map_x - 0.5f * obstacle.size_x;
            const float y_limit = 0.5f * cfg.map_y - 0.5f * obstacle.size_y;
            obstacle.position.x() = uniform(rng, -x_limit, x_limit);
            obstacle.position.y() = uniform(rng, -y_limit, y_limit);

            accepted = true;
            for (const auto &existing : obstacles)
            {
                if (boxesOverlap(obstacle, existing, cfg.obstacle_spawn_clearance))
                {
                    accepted = false;
                    break;
                }
            }
        }
        if (!accepted)
            throw std::runtime_error("Unable to place moving obstacles at the requested density");

        if (obstacle.is_dynamic)
        {
            const float speed =
                uniform(rng, cfg.obstacle_speed_min, cfg.obstacle_speed_max);
            const float angle = uniform(rng, -kPi, kPi);
            obstacle.velocity =
                speed * Eigen::Vector2f(std::cos(angle), std::sin(angle));
        }
        else
            obstacle.velocity.setZero();
        obstacles.push_back(obstacle);
    }
    return obstacles;
}

void advanceObstacles(std::vector<MovingObstacle> &obstacles,
                      const GeneratorConfig &cfg,
                      const Eigen::Vector3f &drone_position,
                      float dt)
{
    for (auto &obstacle : obstacles)
    {
        if (!obstacle.is_dynamic)
            continue;
        const Eigen::Vector2f old_position = obstacle.position;
        obstacle.position += obstacle.velocity * dt;
        const float x_limit = 0.5f * cfg.map_x - 0.5f * obstacle.size_x;
        const float y_limit = 0.5f * cfg.map_y - 0.5f * obstacle.size_y;

        if (obstacle.position.x() < -x_limit || obstacle.position.x() > x_limit)
        {
            obstacle.position.x() = std::clamp(obstacle.position.x(), -x_limit, x_limit);
            obstacle.velocity.x() = -obstacle.velocity.x();
        }
        if (obstacle.position.y() < -y_limit || obstacle.position.y() > y_limit)
        {
            obstacle.position.y() = std::clamp(obstacle.position.y(), -y_limit, y_limit);
            obstacle.velocity.y() = -obstacle.velocity.y();
        }

        // Reflect at the first contact point and use the remaining fraction of
        // the timestep to move away. This avoids both penetration and the
        // two-position oscillation caused by reversing a complete timestep.
        if (cfg.reflect_obstacles_at_drone &&
            signedDistanceToBox(drone_position, obstacle, cfg.drone_radius) <
            cfg.obstacle_drone_reflection_margin)
        {
            const Eigen::Vector2f incoming_velocity = obstacle.velocity;
            float safe_fraction = 0.0f;
            float unsafe_fraction = 1.0f;
            for (int iteration = 0; iteration < 12; ++iteration)
            {
                const float middle = 0.5f * (safe_fraction + unsafe_fraction);
                obstacle.position =
                    old_position + incoming_velocity * (dt * middle);
                if (signedDistanceToBox(drone_position, obstacle, cfg.drone_radius) >=
                    cfg.obstacle_drone_reflection_margin)
                    safe_fraction = middle;
                else
                    unsafe_fraction = middle;
            }

            obstacle.velocity = -incoming_velocity;
            obstacle.position =
                old_position + incoming_velocity * (dt * safe_fraction) +
                obstacle.velocity * (dt * (1.0f - safe_fraction));
            obstacle.position.x() =
                std::clamp(obstacle.position.x(), -x_limit, x_limit);
            obstacle.position.y() =
                std::clamp(obstacle.position.y(), -y_limit, y_limit);

            if (signedDistanceToBox(drone_position, obstacle, cfg.drone_radius) < 0.0f)
                obstacle.position = old_position;
        }
    }
}

Eigen::Vector3f sampleFreePoint(const GeneratorConfig &cfg,
                                const std::vector<MovingObstacle> &obstacles,
                                std::mt19937 &rng,
                                float required_clearance)
{
    for (int attempt = 0; attempt < 20000; ++attempt)
    {
        Eigen::Vector3f point;
        point.x() = uniform(rng, -0.5f * cfg.map_x + cfg.boundary_margin,
                            0.5f * cfg.map_x - cfg.boundary_margin);
        point.y() = uniform(rng, -0.5f * cfg.map_y + cfg.boundary_margin,
                            0.5f * cfg.map_y - cfg.boundary_margin);
        point.z() = uniform(rng, cfg.drone_z_min, cfg.drone_z_max);
        if (minimumClearance(point, obstacles, cfg.drone_radius) > required_clearance)
            return point;
    }
    throw std::runtime_error("Unable to sample a collision-free drone state");
}

DroneState generateDroneState(const GeneratorConfig &cfg,
                              const std::vector<MovingObstacle> &obstacles,
                              std::mt19937 &rng)
{
    DroneState state;
    state.position = sampleFreePoint(cfg, obstacles, rng, cfg.drone_safe_distance);
    state.goal = sampleFreePoint(cfg, obstacles, rng, cfg.drone_safe_distance);
    while ((state.goal - state.position).head<2>().norm() < 0.25f * std::min(cfg.map_x, cfg.map_y))
        state.goal = sampleFreePoint(cfg, obstacles, rng, cfg.drone_safe_distance);

    state.target_speed = uniform(rng, cfg.drone_speed_min, cfg.drone_speed_max);
    state.velocity = (state.goal - state.position).normalized() * state.target_speed;
    return state;
}

Eigen::Quaternionf bodyOrientation(const DroneState &state)
{
    const float horizontal_speed = state.velocity.head<2>().norm();
    const float yaw = horizontal_speed > 1e-4f
                          ? std::atan2(state.velocity.y(), state.velocity.x())
                          : 0.0f;
    const float pitch = -std::atan2(state.velocity.z(), std::max(horizontal_speed, 1e-4f));
    const float lateral_acceleration =
        -std::sin(yaw) * state.acceleration.x() + std::cos(yaw) * state.acceleration.y();
    const float roll = std::clamp(-std::atan2(lateral_acceleration, 9.81f),
                                  -20.0f * kPi / 180.0f,
                                  20.0f * kPi / 180.0f);

    return Eigen::AngleAxisf(yaw, Eigen::Vector3f::UnitZ()) *
           Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY()) *
           Eigen::AngleAxisf(roll, Eigen::Vector3f::UnitX());
}

void chooseNewGoal(DroneState &state,
                   const GeneratorConfig &cfg,
                   const std::vector<MovingObstacle> &obstacles,
                   std::mt19937 &rng)
{
    state.goal = sampleFreePoint(cfg, obstacles, rng, cfg.drone_safe_distance);
    state.target_speed = uniform(rng, cfg.drone_speed_min, cfg.drone_speed_max);
}

void advanceDrone(DroneState &state,
                  const GeneratorConfig &cfg,
                  const std::vector<MovingObstacle> &obstacles,
                  std::mt19937 &rng,
                  float dt)
{
    if ((state.goal - state.position).norm() < cfg.waypoint_tolerance)
        chooseNewGoal(state, cfg, obstacles, rng);

    Eigen::Vector3f goal_direction = state.goal - state.position;
    Eigen::Vector3f desired_velocity = Eigen::Vector3f::Zero();
    if (goal_direction.norm() > 1e-5f)
        desired_velocity = goal_direction.normalized() * state.target_speed;

    Eigen::Vector3f acceleration =
        (desired_velocity - state.velocity) / cfg.velocity_time_constant;

    for (const auto &obstacle : obstacles)
    {
        const float half_x = 0.5f * obstacle.size_x + cfg.drone_radius;
        const float half_y = 0.5f * obstacle.size_y + cfg.drone_radius;
        const Eigen::Vector2f relative =
            state.position.head<2>() - obstacle.position;
        const Eigen::Vector2f closest(
            std::clamp(relative.x(), -half_x, half_x),
            std::clamp(relative.y(), -half_y, half_y));
        Eigen::Vector2f separation = relative - closest;
        float distance = separation.norm();

        if (distance < 1e-4f)
        {
            separation = relative.norm() > 1e-4f
                             ? relative.normalized()
                             : Eigen::Vector2f(1.0f, 0.0f);
            distance = 0.0f;
        }
        else
            separation /= distance;

        if (distance < cfg.avoidance_radius)
        {
            const float ratio =
                (cfg.avoidance_radius - distance) / cfg.avoidance_radius;
            acceleration.head<2>() += cfg.avoidance_gain * ratio * ratio * separation;
        }
    }

    const float x_limit = 0.5f * cfg.map_x - cfg.boundary_margin;
    const float y_limit = 0.5f * cfg.map_y - cfg.boundary_margin;
    const float boundary_gain = 8.0f;
    if (state.position.x() < -x_limit + cfg.boundary_margin)
        acceleration.x() += boundary_gain;
    if (state.position.x() > x_limit - cfg.boundary_margin)
        acceleration.x() -= boundary_gain;
    if (state.position.y() < -y_limit + cfg.boundary_margin)
        acceleration.y() += boundary_gain;
    if (state.position.y() > y_limit - cfg.boundary_margin)
        acceleration.y() -= boundary_gain;

    clampNorm(acceleration, cfg.max_acceleration);
    const Eigen::Vector3f old_position = state.position;
    state.velocity += acceleration * dt;
    clampNorm(state.velocity, cfg.drone_speed_max);
    state.position += state.velocity * dt;
    state.acceleration = acceleration;

    state.position.x() = std::clamp(state.position.x(), -x_limit, x_limit);
    state.position.y() = std::clamp(state.position.y(), -y_limit, y_limit);
    state.position.z() = std::clamp(state.position.z(), cfg.drone_z_min, cfg.drone_z_max);

    if (minimumClearance(state.position, obstacles, cfg.drone_radius) < 0.0f)
    {
        state.position = old_position;
        state.velocity *= 0.2f;
        chooseNewGoal(state, cfg, obstacles, rng);
    }
}

std::vector<DynamicBox> makeBoxes(const std::vector<MovingObstacle> &obstacles)
{
    std::vector<DynamicBox> boxes;
    boxes.reserve(obstacles.size());
    for (const auto &obstacle : obstacles)
        boxes.push_back(obstacle.box());
    return boxes;
}

std::vector<DynamicBox> makeStaticBoxes(
    const std::vector<MovingObstacle> &obstacles)
{
    std::vector<DynamicBox> boxes;
    boxes.reserve(obstacles.size());
    for (const auto &obstacle : obstacles)
    {
        if (!obstacle.is_dynamic)
            boxes.push_back(obstacle.box());
    }
    return boxes;
}

void writeMetadata(const fs::path &episode_path,
                   const GeneratorConfig &cfg,
                   int episode,
                   int obstacle_count,
                   int dynamic_count,
                   int future_frames)
{
    std::ofstream metadata(episode_path / "metadata.yaml");
    metadata << "schema_version: 1\n";
    metadata << "episode: " << episode << "\n";
    metadata << "seed: " << cfg.seed + episode << "\n";
    metadata << "frames: " << cfg.frames_per_episode << "\n";
    metadata << "frame_rate: " << cfg.frame_rate << "\n";
    metadata << "history_length: " << cfg.history_length << "\n";
    metadata << "future_horizon: " << cfg.future_horizon << "\n";
    metadata << "future_frames: " << future_frames << "\n";
    metadata << "valid_frame_start: " << cfg.history_length - 1 << "\n";
    metadata << "valid_frame_end: "
             << std::max(-1, cfg.frames_per_episode - future_frames - 1) << "\n";
    metadata << "map_size: [" << cfg.map_x << ", " << cfg.map_y << ", "
             << cfg.map_z << "]\n";
    metadata << "obstacle_count: " << obstacle_count << "\n";
    metadata << "scenario: " << cfg.scenario << "\n";
    metadata << "dynamic_obstacle_count: " << dynamic_count << "\n";
    metadata << "static_obstacle_count: " << obstacle_count - dynamic_count << "\n";
    metadata << "dynamic_obstacle_ratio: " << cfg.dynamic_ratio << "\n";
    metadata << "requested_area_per_obstacle: " << cfg.area_per_obstacle << "\n";
    metadata << "actual_area_per_obstacle: "
             << cfg.map_x * cfg.map_y / obstacle_count << "\n";
    metadata << "obstacle_speed_range: [" << cfg.obstacle_speed_min << ", "
             << cfg.obstacle_speed_max << "]\n";
    metadata << "obstacle_shape: axis_aligned_box\n";
    metadata << "reflect_obstacles_at_drone: "
             << (cfg.reflect_obstacles_at_drone ? "true" : "false") << "\n";
    metadata << "obstacle_drone_reflection_margin: "
             << cfg.obstacle_drone_reflection_margin << "\n";
    metadata << "static_geometry: ground_plane\n";
    metadata << "all_obstacles_moving: "
             << (dynamic_count == obstacle_count ? "true" : "false") << "\n";
    metadata << "obstacle_tensor_file: obstacles.bin\n";
    metadata << "obstacle_tensor_layout: "
                "frame_obstacle_[cx_cy_cz_halfx_halfy_halfz]_float32\n";
    metadata << "camera:\n";
    metadata << "  fx: " << cfg.camera.fx << "\n";
    metadata << "  fy: " << cfg.camera.fy << "\n";
    metadata << "  cx: " << cfg.camera.cx << "\n";
    metadata << "  cy: " << cfg.camera.cy << "\n";
    metadata << "  width: " << cfg.camera.image_width << "\n";
    metadata << "  height: " << cfg.camera.image_height << "\n";
    metadata << "  max_depth: " << cfg.camera.max_depth_dist << "\n";
    metadata << "depth_encoding: uint16_normalized_0_65535\n";
    metadata << "dynamic_mask_folder: dynamic_mask\n";
    metadata << "dynamic_mask_encoding: uint8_0_static_255_dynamic\n";
    metadata << "relative_pose_convention: T_camera_previous_camera_current\n";
}

void printProgress(int episode, int frame, int total)
{
    const int interval = std::max(1, total / 100);
    if ((frame + 1) % interval == 0 || frame + 1 == total)
    {
        const float percentage = 100.0f * static_cast<float>(frame + 1) /
                                 static_cast<float>(total);
        std::cout << "\rEpisode " << episode << ": " << std::fixed
                  << std::setprecision(1) << percentage << "%" << std::flush;
    }
}

void generateEpisode(const GeneratorConfig &cfg,
                     int episode,
                     DynamicDepthRenderer &renderer,
                     std::ofstream &manifest)
{
    std::mt19937 rng(cfg.seed + episode);
    std::vector<MovingObstacle> obstacles = generateObstacles(cfg, rng);
    DroneState drone = generateDroneState(cfg, obstacles, rng);

    std::ostringstream episode_name;
    episode_name << "episode_" << std::setw(4) << std::setfill('0') << episode;
    const fs::path episode_path = cfg.save_path / episode_name.str();
    const fs::path depth_path = episode_path / "depth";
    const fs::path dynamic_mask_path = episode_path / "dynamic_mask";
    fs::create_directories(depth_path);
    fs::create_directories(dynamic_mask_path);

    std::ofstream drone_file(episode_path / "drone_state.csv");
    std::ofstream relative_file(episode_path / "relative_pose.csv");
    std::ofstream obstacle_file(episode_path / "obstacles.csv");
    std::ofstream obstacle_binary(episode_path / "obstacles.bin",
                                  std::ios::out | std::ios::binary);

    drone_file << "frame,t,valid,px,py,pz,body_qw,body_qx,body_qy,body_qz,"
                  "camera_qw,camera_qx,camera_qy,camera_qz,vx,vy,vz,ax,ay,az,"
                  "goal_x,goal_y,goal_z,collision,min_clearance\n";
    relative_file << "frame,t,dt,tx,ty,tz,qw,qx,qy,qz\n";
    obstacle_file << "frame,t,id,is_dynamic,cx,cy,cz,vx,vy,vz,size_x,size_y,height\n";

    const float dt = 1.0f / cfg.frame_rate;
    const int future_frames =
        static_cast<int>(std::ceil(cfg.future_horizon * cfg.frame_rate));
    const Eigen::Quaternionf q_bc(
        Eigen::AngleAxisf(cfg.camera_pitch_rad, Eigen::Vector3f::UnitY()));

    Eigen::Vector3f previous_position = drone.position;
    Eigen::Quaternionf previous_q_wc = bodyOrientation(drone) * q_bc;
    float minimum_episode_clearance = std::numeric_limits<float>::max();
    int collision_frames = 0;

    for (int frame = 0; frame < cfg.frames_per_episode; ++frame)
    {
        const float time = frame * dt;
        const Eigen::Quaternionf q_wb = bodyOrientation(drone);
        const Eigen::Quaternionf q_wc = q_wb * q_bc;
        const std::vector<DynamicBox> boxes = makeBoxes(obstacles);
        const std::vector<DynamicBox> static_boxes = makeStaticBoxes(obstacles);
        const float clearance =
            minimumClearance(drone.position, obstacles, cfg.drone_radius);
        const bool collision = clearance <= 0.0f;
        minimum_episode_clearance = std::min(minimum_episode_clearance, clearance);
        collision_frames += collision ? 1 : 0;

        cudaMat::SE3<float> T_wc(q_wc.w(), q_wc.x(), q_wc.y(), q_wc.z(),
                                 drone.position.x(), drone.position.y(), drone.position.z());
        cv::Mat depth;
        cv::Mat static_depth;
        renderer.render(boxes, T_wc, depth);
        renderer.render(static_boxes, T_wc, static_depth);

        std::ostringstream image_name;
        image_name << "img_" << std::setw(6) << std::setfill('0') << frame << ".png";
        saveDepth16(depth, cfg.camera.max_depth_dist, depth_path / image_name.str());
        saveDynamicMask(depth, static_depth, dynamic_mask_path / image_name.str());

        const bool valid = frame >= cfg.history_length - 1 &&
                           frame + future_frames < cfg.frames_per_episode;
        drone_file << std::fixed << std::setprecision(7)
                   << frame << "," << time << "," << static_cast<int>(valid) << ","
                   << drone.position.x() << "," << drone.position.y() << ","
                   << drone.position.z() << ","
                   << q_wb.w() << "," << q_wb.x() << "," << q_wb.y() << ","
                   << q_wb.z() << ","
                   << q_wc.w() << "," << q_wc.x() << "," << q_wc.y() << ","
                   << q_wc.z() << ","
                   << drone.velocity.x() << "," << drone.velocity.y() << ","
                   << drone.velocity.z() << ","
                   << drone.acceleration.x() << "," << drone.acceleration.y() << ","
                   << drone.acceleration.z() << ","
                   << drone.goal.x() << "," << drone.goal.y() << "," << drone.goal.z()
                   << "," << static_cast<int>(collision) << "," << clearance << "\n";

        Eigen::Vector3f relative_translation = Eigen::Vector3f::Zero();
        Eigen::Quaternionf relative_rotation = Eigen::Quaternionf::Identity();
        if (frame > 0)
        {
            relative_translation =
                previous_q_wc.conjugate() * (drone.position - previous_position);
            relative_rotation = previous_q_wc.conjugate() * q_wc;
            relative_rotation.normalize();
        }
        relative_file << std::fixed << std::setprecision(7)
                      << frame << "," << time << "," << (frame > 0 ? dt : 0.0f) << ","
                      << relative_translation.x() << "," << relative_translation.y() << ","
                      << relative_translation.z() << ","
                      << relative_rotation.w() << "," << relative_rotation.x() << ","
                      << relative_rotation.y() << "," << relative_rotation.z() << "\n";

        for (const auto &obstacle : obstacles)
        {
            obstacle_file << std::fixed << std::setprecision(7)
                          << frame << "," << time << "," << obstacle.id << ","
                          << static_cast<int>(obstacle.is_dynamic) << ","
                          << obstacle.position.x() << "," << obstacle.position.y() << ","
                          << 0.5f * obstacle.height << ","
                          << obstacle.velocity.x() << "," << obstacle.velocity.y() << ",0,"
                          << obstacle.size_x << "," << obstacle.size_y << ","
                          << obstacle.height << "\n";

            const float obstacle_tensor[6] = {
                obstacle.position.x(), obstacle.position.y(),
                0.5f * obstacle.height, 0.5f * obstacle.size_x,
                0.5f * obstacle.size_y, 0.5f * obstacle.height};
            obstacle_binary.write(
                reinterpret_cast<const char *>(obstacle_tensor),
                sizeof(obstacle_tensor));
        }

        previous_position = drone.position;
        previous_q_wc = q_wc;
        printProgress(episode, frame, cfg.frames_per_episode);

        if (frame + 1 < cfg.frames_per_episode)
        {
            advanceObstacles(obstacles, cfg, drone.position, dt);
            advanceDrone(drone, cfg, obstacles, rng, dt);
        }
    }

    const int dynamic_count = static_cast<int>(std::count_if(
        obstacles.begin(), obstacles.end(),
        [](const MovingObstacle &obstacle) { return obstacle.is_dynamic; }));
    writeMetadata(episode_path, cfg, episode, static_cast<int>(obstacles.size()),
                  dynamic_count, future_frames);
    manifest << episode_name.str() << "," << cfg.seed + episode << ","
             << cfg.frames_per_episode << "," << cfg.scenario << ","
             << obstacles.size() << "," << dynamic_count << ","
             << minimum_episode_clearance << "," << collision_frames << "\n";
    std::cout << "  min_clearance=" << minimum_episode_clearance
              << " m, collision_frames=" << collision_frames << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    try
    {
        GeneratorConfig cfg = loadConfig();
        applyArguments(argc, argv, cfg);
        validateConfig(cfg);
        prepareSavePath(cfg.save_path);

        const int obstacle_count = std::max(
            1, static_cast<int>(std::lround(cfg.map_x * cfg.map_y /
                                            cfg.area_per_obstacle)));
        const int future_frames =
            static_cast<int>(std::ceil(cfg.future_horizon * cfg.frame_rate));

        std::cout << "Temporal box dataset generator" << std::endl;
        std::cout << "  output: " << cfg.save_path << std::endl;
        std::cout << "  episodes: " << cfg.episode_num
                  << ", frames/episode: " << cfg.frames_per_episode << std::endl;
        std::cout << "  map: " << cfg.map_x << " x " << cfg.map_y
                  << " m, obstacles: " << obstacle_count
                  << " (1 per " << cfg.map_x * cfg.map_y / obstacle_count
                  << " m^2)" << std::endl;
        std::cout << "  obstacle speed: [" << cfg.obstacle_speed_min << ", "
                  << cfg.obstacle_speed_max << "] m/s" << std::endl;
        std::cout << "  scenario: " << cfg.scenario
                  << ", dynamic ratio: " << cfg.dynamic_ratio << std::endl;
        std::cout << "  history: " << cfg.history_length
                  << " frames, future labels: " << future_frames << " frames"
                  << std::endl;

        std::ofstream manifest(cfg.save_path / "manifest.csv");
        manifest << "episode,seed,frames,scenario,obstacle_count,dynamic_count,"
                    "min_clearance,collision_frames\n";

        DynamicDepthRenderer renderer(cfg.camera, obstacle_count);
        const auto start = std::chrono::steady_clock::now();
        for (int episode = 0; episode < cfg.episode_num; ++episode)
            generateEpisode(cfg, episode, renderer, manifest);
        const double elapsed = std::chrono::duration<double>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();

        std::cout << "Dataset generation completed in " << elapsed << " s ("
                  << cfg.episode_num * cfg.frames_per_episode / elapsed
                  << " frames/s)." << std::endl;
        return 0;
    }
    catch (const std::exception &error)
    {
        std::cerr << "dynamic_dataset_generator failed: " << error.what() << std::endl;
        return 1;
    }
}
