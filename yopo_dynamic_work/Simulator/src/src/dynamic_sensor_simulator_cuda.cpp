#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Odometry.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/Bool.h>
#include <yaml-cpp/yaml.h>

#include "sensor_simulator.cuh"

using raycast::CameraParams;
using raycast::DynamicBox;
using raycast::DynamicDepthRenderer;

namespace
{
constexpr float kPi = 3.14159265358979323846f;

struct OnlineObstacle
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

float uniform(std::mt19937 &rng, float lower, float upper)
{
    std::uniform_real_distribution<float> distribution(lower, upper);
    return distribution(rng);
}

bool boxesOverlap(const OnlineObstacle &a,
                  const OnlineObstacle &b,
                  float clearance)
{
    return std::abs(a.position.x() - b.position.x()) <
               0.5f * (a.size_x + b.size_x) + clearance &&
           std::abs(a.position.y() - b.position.y()) <
               0.5f * (a.size_y + b.size_y) + clearance;
}

float pointBoxDistance2D(const Eigen::Vector2f &point,
                         const OnlineObstacle &obstacle)
{
    const Eigen::Vector2f q =
        (point - obstacle.position).cwiseAbs() -
        Eigen::Vector2f(0.5f * obstacle.size_x, 0.5f * obstacle.size_y);
    return q.cwiseMax(0.0f).norm() +
           std::min(std::max(q.x(), q.y()), 0.0f);
}

float signedDistanceToBox(const Eigen::Vector3f &point,
                          const OnlineObstacle &obstacle,
                          float drone_radius)
{
    const Eigen::Vector3f center(
        obstacle.position.x(), obstacle.position.y(), 0.5f * obstacle.height);
    const Eigen::Vector3f half(
        0.5f * obstacle.size_x, 0.5f * obstacle.size_y,
        0.5f * obstacle.height);
    const Eigen::Vector3f q = (point - center).cwiseAbs() - half;
    return q.cwiseMax(0.0f).norm() +
           std::min(std::max(q.x(), std::max(q.y(), q.z())), 0.0f) -
           drone_radius;
}
}  // namespace

class DynamicSensorSimulator
{
public:
    explicit DynamicSensorSimulator(ros::NodeHandle &node) : node_(node)
    {
        const YAML::Node config = YAML::LoadFile(CONFIG_FILE_PATH);
        const YAML::Node dataset = config["dynamic_dataset"];
        const YAML::Node online = config["online_dynamic"];
        if (!dataset || !online)
            throw std::runtime_error(
                "dynamic_dataset and online_dynamic config sections are required");

        camera_.fx = config["camera"]["fx"].as<float>();
        camera_.fy = config["camera"]["fy"].as<float>();
        camera_.cx = config["camera"]["cx"].as<float>();
        camera_.cy = config["camera"]["cy"].as<float>();
        camera_.image_width = config["camera"]["image_width"].as<int>();
        camera_.image_height = config["camera"]["image_height"].as<int>();
        camera_.max_depth_dist =
            config["camera"]["max_depth_dist"].as<float>();
        camera_.normalize_depth = false;
        const float pitch =
            config["camera"]["pitch"].as<float>() * kPi / 180.0f;
        q_bc_ = Eigen::AngleAxisf(pitch, Eigen::Vector3f::UnitY());

        map_x_ = config["x_length"].as<float>();
        map_y_ = config["y_length"].as<float>();
        area_per_obstacle_ = dataset["area_per_obstacle"].as<float>();
        width_min_ = dataset["obstacle_width_min"].as<float>();
        width_max_ = dataset["obstacle_width_max"].as<float>();
        height_min_ = dataset["obstacle_height_min"].as<float>();
        height_max_ = dataset["obstacle_height_max"].as<float>();
        speed_min_ = dataset["obstacle_speed_min"].as<float>();
        speed_max_ = dataset["obstacle_speed_max"].as<float>();
        spawn_clearance_ = dataset["obstacle_spawn_clearance"].as<float>();
        drone_radius_ = dataset["drone_radius"].as<float>();

        seed_ = online["seed"].as<int>();
        dynamic_ratio_ = online["dynamic_ratio"].as<float>();
        origin_clearance_ = online["origin_clearance"].as<float>();
        if (dynamic_ratio_ < 0.0f || dynamic_ratio_ > 1.0f)
            throw std::runtime_error("online_dynamic.dynamic_ratio must be in [0,1]");

        const int obstacle_count = std::max(
            1, static_cast<int>(std::lround(map_x_ * map_y_ /
                                            area_per_obstacle_)));
        generateObstacles(obstacle_count);
        renderer_ = std::make_unique<DynamicDepthRenderer>(
            camera_, obstacle_count);

        const float depth_fps = config["depth_fps"].as<float>();
        depth_period_ = ros::Duration(1.0 / depth_fps);
        const std::string odom_topic = config["odom_topic"].as<std::string>();
        const std::string depth_topic =
            config["depth_topic"].as<std::string>();
        const std::string marker_topic =
            online["marker_topic"].as<std::string>();
        const std::string collision_topic =
            online["collision_topic"].as<std::string>();

        image_publisher_ =
            node_.advertise<sensor_msgs::Image>(depth_topic, 1);
        marker_publisher_ =
            node_.advertise<sensor_msgs::PointCloud2>(marker_topic, 1);
        collision_publisher_ =
            node_.advertise<std_msgs::Bool>(collision_topic, 1);
        odom_subscriber_ = node_.subscribe(
            odom_topic, 1, &DynamicSensorSimulator::odomCallback, this,
            ros::TransportHints().tcpNoDelay());

        ROS_INFO(
            "Dynamic CUDA simulator ready: %d boxes, %.0f dynamic, "
            "speed %.1f-%.1f m/s",
            obstacle_count, dynamic_ratio_ * obstacle_count,
            speed_min_, speed_max_);
    }

private:
    void generateObstacles(int obstacle_count)
    {
        std::mt19937 rng(seed_);
        const int dynamic_count = static_cast<int>(
            std::lround(dynamic_ratio_ * obstacle_count));
        std::vector<int> flags(obstacle_count, 0);
        std::fill(flags.begin(), flags.begin() + dynamic_count, 1);
        std::shuffle(flags.begin(), flags.end(), rng);

        obstacles_.reserve(obstacle_count);
        for (int id = 0; id < obstacle_count; ++id)
        {
            OnlineObstacle obstacle;
            obstacle.id = id;
            obstacle.is_dynamic = flags[id] != 0;
            bool accepted = false;
            for (int attempt = 0; attempt < 20000 && !accepted; ++attempt)
            {
                obstacle.size_x = uniform(rng, width_min_, width_max_);
                obstacle.size_y = uniform(rng, width_min_, width_max_);
                obstacle.height = uniform(rng, height_min_, height_max_);
                const float x_limit =
                    0.5f * map_x_ - 0.5f * obstacle.size_x;
                const float y_limit =
                    0.5f * map_y_ - 0.5f * obstacle.size_y;
                obstacle.position.x() = uniform(rng, -x_limit, x_limit);
                obstacle.position.y() = uniform(rng, -y_limit, y_limit);

                accepted =
                    pointBoxDistance2D(Eigen::Vector2f::Zero(), obstacle) >=
                    origin_clearance_;
                for (const auto &existing : obstacles_)
                {
                    if (accepted &&
                        boxesOverlap(obstacle, existing, spawn_clearance_))
                        accepted = false;
                }
            }
            if (!accepted)
                throw std::runtime_error(
                    "Unable to place online dynamic obstacles");

            if (obstacle.is_dynamic)
            {
                const float speed = uniform(rng, speed_min_, speed_max_);
                const float angle = uniform(rng, -kPi, kPi);
                obstacle.velocity = speed * Eigen::Vector2f(
                    std::cos(angle), std::sin(angle));
            }
            obstacles_.push_back(obstacle);
        }
    }

    void advanceObstacles(float dt)
    {
        if (!(dt > 0.0f))
            return;
        const int substeps =
            std::max(1, static_cast<int>(std::ceil(dt / 0.02f)));
        const float sub_dt = dt / substeps;
        for (int step = 0; step < substeps; ++step)
        {
            for (auto &obstacle : obstacles_)
            {
                if (!obstacle.is_dynamic)
                    continue;
                obstacle.position += obstacle.velocity * sub_dt;
                const float x_limit =
                    0.5f * map_x_ - 0.5f * obstacle.size_x;
                const float y_limit =
                    0.5f * map_y_ - 0.5f * obstacle.size_y;
                if (obstacle.position.x() < -x_limit ||
                    obstacle.position.x() > x_limit)
                {
                    obstacle.position.x() =
                        std::clamp(obstacle.position.x(), -x_limit, x_limit);
                    obstacle.velocity.x() = -obstacle.velocity.x();
                }
                if (obstacle.position.y() < -y_limit ||
                    obstacle.position.y() > y_limit)
                {
                    obstacle.position.y() =
                        std::clamp(obstacle.position.y(), -y_limit, y_limit);
                    obstacle.velocity.y() = -obstacle.velocity.y();
                }
            }
        }
    }

    std::vector<DynamicBox> boxes() const
    {
        std::vector<DynamicBox> result;
        result.reserve(obstacles_.size());
        for (const auto &obstacle : obstacles_)
            result.push_back(obstacle.box());
        return result;
    }

    void publishObstacleCloud(const ros::Time &stamp)
    {
        if (marker_publisher_.getNumSubscribers() == 0)
            return;
        pcl::PointCloud<pcl::PointXYZI> cloud;
        cloud.header.frame_id = "world";
        const float resolution = 0.25f;
        for (const auto &obstacle : obstacles_)
        {
            const float half_x = 0.5f * obstacle.size_x;
            const float half_y = 0.5f * obstacle.size_y;
            const float intensity = obstacle.is_dynamic ?
                obstacle.velocity.norm() : 0.0f;
            auto appendPoint = [&](float x, float y, float z)
            {
                pcl::PointXYZI point;
                point.x = x;
                point.y = y;
                point.z = z;
                point.intensity = intensity;
                cloud.points.push_back(point);
            };
            for (float z = 0.0f; z <= obstacle.height; z += resolution)
            {
                for (float x = -half_x; x <= half_x; x += resolution)
                {
                    appendPoint(obstacle.position.x() + x,
                                obstacle.position.y() - half_y, z);
                    appendPoint(obstacle.position.x() + x,
                                obstacle.position.y() + half_y, z);
                }
                for (float y = -half_y; y <= half_y; y += resolution)
                {
                    appendPoint(obstacle.position.x() - half_x,
                                obstacle.position.y() + y, z);
                    appendPoint(obstacle.position.x() + half_x,
                                obstacle.position.y() + y, z);
                }
            }
        }
        cloud.width = static_cast<uint32_t>(cloud.points.size());
        cloud.height = 1;
        cloud.is_dense = true;
        sensor_msgs::PointCloud2 message;
        pcl::toROSMsg(cloud, message);
        message.header.frame_id = "world";
        message.header.stamp = stamp;
        marker_publisher_.publish(message);
    }

    void renderAndPublish(const ros::Time &stamp)
    {
        const cudaMat::SE3<float> T_wc(
            q_wc_.w(), q_wc_.x(), q_wc_.y(), q_wc_.z(),
            drone_position_.x(), drone_position_.y(), drone_position_.z());
        cv::Mat depth;
        renderer_->render(boxes(), T_wc, depth);

        cv_bridge::CvImage image;
        image.header.frame_id = "camera";
        image.header.stamp = stamp;
        image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        image.image = depth;
        image_publisher_.publish(image.toImageMsg());
        publishObstacleCloud(stamp);

        bool collision = false;
        for (const auto &obstacle : obstacles_)
        {
            if (signedDistanceToBox(
                    drone_position_, obstacle, drone_radius_) <= 0.0f)
            {
                collision = true;
                break;
            }
        }
        std_msgs::Bool collision_message;
        collision_message.data = collision;
        collision_publisher_.publish(collision_message);
        if (collision)
            ROS_WARN_THROTTLE(1.0, "Dynamic obstacle collision detected");
    }

    void odomCallback(const nav_msgs::Odometry::ConstPtr &message)
    {
        const ros::Time stamp =
            message->header.stamp.isZero() ?
                ros::Time::now() : message->header.stamp;
        if (!last_obstacle_stamp_.isZero())
        {
            const double dt = (stamp - last_obstacle_stamp_).toSec();
            if (dt > 0.0 && dt < 0.5)
                advanceObstacles(static_cast<float>(dt));
        }
        last_obstacle_stamp_ = stamp;

        q_wb_.x() = message->pose.pose.orientation.x;
        q_wb_.y() = message->pose.pose.orientation.y;
        q_wb_.z() = message->pose.pose.orientation.z;
        q_wb_.w() = message->pose.pose.orientation.w;
        q_wc_ = q_wb_ * q_bc_;
        drone_position_.x() = message->pose.pose.position.x;
        drone_position_.y() = message->pose.pose.position.y;
        drone_position_.z() = message->pose.pose.position.z;

        const ros::Time now = ros::Time::now();
        if (next_depth_time_.isZero() ||
            std::abs((now - next_depth_time_).toSec()) >
                10.0 * depth_period_.toSec())
            next_depth_time_ = now;
        if (now >= next_depth_time_)
        {
            do
            {
                next_depth_time_ += depth_period_;
            } while (next_depth_time_ <= now);
            renderAndPublish(stamp);
        }
    }

    ros::NodeHandle node_;
    ros::Publisher image_publisher_;
    ros::Publisher marker_publisher_;
    ros::Publisher collision_publisher_;
    ros::Subscriber odom_subscriber_;

    CameraParams camera_;
    std::unique_ptr<DynamicDepthRenderer> renderer_;
    std::vector<OnlineObstacle> obstacles_;
    Eigen::Quaternionf q_wb_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf q_bc_{Eigen::Quaternionf::Identity()};
    Eigen::Quaternionf q_wc_{Eigen::Quaternionf::Identity()};
    Eigen::Vector3f drone_position_{Eigen::Vector3f::Zero()};

    float map_x_{60.0f};
    float map_y_{60.0f};
    float area_per_obstacle_{30.0f};
    float width_min_{0.6f};
    float width_max_{1.5f};
    float height_min_{3.0f};
    float height_max_{6.0f};
    float speed_min_{0.5f};
    float speed_max_{3.0f};
    float spawn_clearance_{0.25f};
    float drone_radius_{0.3f};
    float dynamic_ratio_{1.0f};
    float origin_clearance_{4.0f};
    int seed_{500};
    ros::Duration depth_period_{1.0 / 33.0};
    ros::Time next_depth_time_;
    ros::Time last_obstacle_stamp_;
};

int main(int argc, char **argv)
{
    ros::init(argc, argv, "dynamic_sensor_simulator_node");
    ros::NodeHandle node;
    try
    {
        DynamicSensorSimulator simulator(node);
        ros::spin();
    }
    catch (const std::exception &error)
    {
        ROS_FATAL("Dynamic sensor simulator failed: %s", error.what());
        return 1;
    }
    return 0;
}
