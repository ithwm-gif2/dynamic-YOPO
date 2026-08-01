#include "sensor_simulator.cuh"

namespace raycast
{   
    namespace
    {
        __device__ bool intersectSlab(float origin, float direction,
                                      float lower, float upper,
                                      float &t_min, float &t_max)
        {
            const float eps = 1e-7f;
            if (fabsf(direction) < eps)
                return origin >= lower && origin <= upper;

            float t1 = (lower - origin) / direction;
            float t2 = (upper - origin) / direction;
            if (t1 > t2)
            {
                const float tmp = t1;
                t1 = t2;
                t2 = tmp;
            }
            t_min = fmaxf(t_min, t1);
            t_max = fminf(t_max, t2);
            return t_max >= t_min;
        }

        __device__ bool intersectDynamicBox(const float3 &origin,
                                            const float3 &direction,
                                            const DynamicBox &box,
                                            float &depth)
        {
            float t_min = -1e30f;
            float t_max = 1e30f;
            if (!intersectSlab(origin.x, direction.x, box.center_x - box.half_x,
                               box.center_x + box.half_x, t_min, t_max) ||
                !intersectSlab(origin.y, direction.y, box.center_y - box.half_y,
                               box.center_y + box.half_y, t_min, t_max) ||
                !intersectSlab(origin.z, direction.z, box.center_z - box.half_z,
                               box.center_z + box.half_z, t_min, t_max))
                return false;

            if (t_max < 0.0f)
                return false;

            depth = t_min >= 0.0f ? t_min : 0.001f;
            return true;
        }

        __global__ void dynamicCameraRaycastKernel(float *depth_values,
                                                   const DynamicBox *boxes,
                                                   int box_count,
                                                   CameraParams camera_param,
                                                   cudaMat::SE3<float> T_wc)
        {
            const int u = threadIdx.x;
            const int v = blockIdx.x;
            if (u >= camera_param.image_width || v >= camera_param.image_height)
                return;

            const float image_y = -(u - camera_param.cx) / camera_param.fx;
            const float image_z = -(v - camera_param.cy) / camera_param.fy;
            const float3 origin_w = T_wc * make_float3(0.0f, 0.0f, 0.0f);
            const float3 ray_point_w = T_wc * make_float3(1.0f, image_y, image_z);
            const float3 direction_w = make_float3(ray_point_w.x - origin_w.x,
                                                   ray_point_w.y - origin_w.y,
                                                   ray_point_w.z - origin_w.z);

            float nearest_depth = camera_param.max_depth_dist;
            if (direction_w.z < -1e-7f)
            {
                const float ground_depth = -origin_w.z / direction_w.z;
                if (ground_depth > 0.0f)
                    nearest_depth = fminf(nearest_depth, ground_depth);
            }

            for (int i = 0; i < box_count; ++i)
            {
                float hit_depth = 0.0f;
                if (intersectDynamicBox(origin_w, direction_w, boxes[i], hit_depth) &&
                    hit_depth < nearest_depth)
                    nearest_depth = hit_depth;
            }

            nearest_depth = fminf(fmaxf(nearest_depth, 0.0f), camera_param.max_depth_dist);
            if (camera_param.normalize_depth)
                nearest_depth /= camera_param.max_depth_dist;
            depth_values[v * camera_param.image_width + u] = nearest_depth;
        }
    }

    GridMap::GridMap(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float resolution, int occupy_threshold = 1){
        const float epsilon = 0.001f;   // 避免数值误差导致 (1)建图空行 (2)边缘点被忽略
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, min_pt, max_pt);
        float length = max_pt(0) - min_pt(0) + 2 * epsilon;  // 保证各个边界最大值能被取到
        float width  = max_pt(1) - min_pt(1) + 2 * epsilon;
        float height = max_pt(2) - min_pt(2) + 2 * epsilon;
        Vector3f origin(min_pt(0), min_pt(1), min_pt(2));
        Vector3f map_size(length, width, height);
        origin_x_ = origin.x;
        origin_y_ = origin.y;
        origin_z_ = origin.z;

        Vector3i grid_size;
        grid_size.x = ceil(map_size.x / resolution);
        grid_size.y = ceil(map_size.y / resolution);
        grid_size.z = ceil(map_size.z / resolution);
        int grid_total_size = grid_size.x * grid_size.y * grid_size.z;

        resolution_   = resolution;
        grid_size_x_  = grid_size.x, 
        grid_size_y_  = grid_size.y, 
        grid_size_z_  = grid_size.z, 
        grid_size_yz_ = grid_size.y * grid_size.z;
        occupy_threshold_ = occupy_threshold;
        raycast_step_ = resolution;

        std::vector<int> h_map(grid_total_size, 0);
        // 点云全位于体素边界，有时候会有全空的行，加个很小的偏移
        for (size_t i = 0; i < cloud->points.size(); i++) {
            Vector3f point(cloud->points[i].x + epsilon, cloud->points[i].y + epsilon, cloud->points[i].z + epsilon);
            int idx = Vox2Idx(Pos2Vox(point));
            if (idx < grid_total_size) {
                h_map[idx]++;
            }
        }
        cudaMalloc((void **)&map_cuda_, grid_total_size * sizeof(int));
        cudaMemcpy(map_cuda_, h_map.data(), grid_total_size * sizeof(int), cudaMemcpyHostToDevice);
    }

    __host__ __device__ Vector3i GridMap::Pos2Vox(const Vector3f &pos)
    {
        Vector3i vox;
        vox.x = floor((pos.x - origin_x_) / resolution_);
        vox.y = floor((pos.y - origin_y_) / resolution_);
        vox.z = floor((pos.z - origin_z_) / resolution_);
        return vox;
    }

    __host__ __device__ Vector3f GridMap::Vox2Pos(const Vector3i &vox)
    {
        Vector3f pos;
        pos.x = (vox.x + 0.5f) * resolution_ + origin_x_;
        pos.y = (vox.y + 0.5f) * resolution_ + origin_y_;
        pos.z = (vox.z + 0.5f) * resolution_ + origin_z_;
        return pos;
    }

    __host__ __device__ int GridMap::Vox2Idx(const Vector3i &vox)
    {
        return vox.x * grid_size_yz_ + vox.y * grid_size_z_ + vox.z;
    }

    __host__ __device__ Vector3i GridMap::Idx2Vox(int idx)
    {
        return Vector3i(idx / grid_size_yz_, (idx % grid_size_yz_) / grid_size_z_, idx % grid_size_z_);
    }

    __device__ int GridMap::symmetricIndex(int index, int length)
    {
        index = index % (2 * length - 2);
        if (index < 0)
        {
            index += (2 * length - 2);
        }

        if (index >= length)
        {
            index = 2 * length - 2 - index;
        }
        return index;
    }

    // -1: z越界; 0: 空闲; 1: 占据
    __device__  int GridMap::mapQuery(const Vector3f &pos){
        Vector3i vox = Pos2Vox(pos);
        vox.x = symmetricIndex(vox.x, grid_size_x_);
        vox.y = symmetricIndex(vox.y, grid_size_y_);

        if (vox.z >= grid_size_z_)
            return 0;
        if (vox.z <= 0)
            return 1;

        int idx = Vox2Idx(vox);
        if (map_cuda_[idx] > occupy_threshold_)
            return 1;
        return 0;        
    }

    __global__ void cameraRaycastKernel(float* depth_values, GridMap grid_map, CameraParams camera_param, cudaMat::SE3<float> T_wc)
    {
        int u = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);

        if (u < camera_param.image_width && v < camera_param.image_height)
        {
            // 计算射线方向
            float y = -(u - camera_param.cx) / camera_param.fx;
            float z = -(v - camera_param.cy) / camera_param.fy;
            float x = 1.0f;

            // 归一化射线方向
            float length = sqrtf(x * x + y * y + z * z);
            x /= length;
            y /= length;
            z /= length;

            // 计算每个轴的增量比例 (x方向固定步长避免近距离处畸变; 0.5是瞎设的防止过于稀疏导致错误)
            float dx = 0.5 * grid_map.raycast_step_;
            float dy = (y / x) * dx;
            float dz = (z / x) * dx;

            // 递增射线方向上的每个轴
            int scale = 0;
            float depth = 0.0f;

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                if (occupied == 1)
                {
                    // depth = point_x;  // 直接这样赋值会有一点误差
                    // 栅格化避免平面变曲面 (有些冗余，但在机体系栅格化会有类似摩尔纹的东西)
                    Vector3i occ_vox_w = grid_map.Pos2Vox(point);
                    Vector3f occ_point_w = grid_map.Vox2Pos(occ_vox_w);
                    float3 occ_point_w_ = make_float3(occ_point_w.x, occ_point_w.y, occ_point_w.z);
                    float3 occ_point_c_ = T_wc.inv() * occ_point_w_;
                    depth = occ_point_c_.x;
                    break;
                }

                if (point_x >= camera_param.max_depth_dist){
                    depth = camera_param.max_depth_dist;
                    break;
                }
            }

            // 将深度值存储到输出数组中
            if (camera_param.normalize_depth)
                depth = depth / camera_param.max_depth_dist;
            depth_values[v * camera_param.image_width + u] = depth;
        }
    }

    void renderDepthImage(GridMap* grid_map, CameraParams* camera_param, cudaMat::SE3<float>& T_wc, cv::Mat& depth_image)
    {   
        float* depth_values;
        size_t num_elements = camera_param->image_width * camera_param->image_height;
        cudaMallocManaged(&depth_values, num_elements * sizeof(float));

        // 在GPU上启动核函数
        cameraRaycastKernel<<<camera_param->image_height, camera_param->image_width>>>(depth_values, *grid_map, *camera_param, T_wc);
        
        cudaDeviceSynchronize();

        depth_image.create(camera_param->image_height, camera_param->image_width, CV_32FC1);

        cudaMemcpy(depth_image.data, depth_values, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
        
        cudaFree(depth_values);
        return;
    }

    __global__ void lidarRaycastKernel(Vector3f* point_values, GridMap grid_map, LidarParams lidar_param, cudaMat::SE3<float> T_wc)
    {
        int h = threadIdx.x;
        int v = blockIdx.x;

        // printf("u: %d, v: %d \n", u, v);
        if (h < lidar_param.horizontal_num && v < lidar_param.vertical_lines)
        {   
            float vertical_resolution = (lidar_param.vertical_angle_end - lidar_param.vertical_angle_start) / (lidar_param.vertical_lines - 1);
            float vertical_angle = lidar_param.vertical_angle_start + v * vertical_resolution;
            float sin_vert = std::sin(vertical_angle * M_PI / 180.0);
            float cos_vert = std::cos(vertical_angle * M_PI / 180.0);
            float horizontal_angle = h * lidar_param.horizontal_resolution;
            float sin_horz = std::sin(horizontal_angle * M_PI / 180.0);
            float cos_horz = std::cos(horizontal_angle * M_PI / 180.0);
            // 计算射线方向
            Vector3f ray_direction(cos_vert * cos_horz, cos_vert * sin_horz, sin_vert);

            // 计算每个轴的增量比例
            float dx = ray_direction.x * grid_map.raycast_step_;
            float dy = ray_direction.y * grid_map.raycast_step_;
            float dz = ray_direction.z * grid_map.raycast_step_;

            // 递增射线方向上的每个轴
            int scale = 0;
            Vector3f point_value(0, 0, 0);

            while (1)
            {
                scale += 1;

                float point_x = scale * dx;
                float point_y = scale * dy;
                float point_z = scale * dz;

                float3 point_c = make_float3(point_x, point_y, point_z);
                float3 point_w = T_wc * point_c;

                Vector3f point(point_w.x, point_w.y, point_w.z);

                int occupied = grid_map.mapQuery(point);

                float ray_length = sqrtf(point_x * point_x + point_y * point_y + point_z * point_z);

                if (occupied == 1)
                {
                    point_value = Vector3f(point_x, point_y, point_z);
                    Vector3i vox_body = grid_map.Pos2Vox(point_value);  // 栅格化避免平面变曲面
                    point_value = grid_map.Vox2Pos(vox_body);
                    break;
                }

                if (ray_length > lidar_param.max_lidar_dist){
                    break;
                }
            }

            // 将点云值存储到输出数组中，(0, 0, 0)为无效值
            point_values[v * lidar_param.horizontal_num + h] = point_value;
        }
    }

    void renderLidarPointcloud(GridMap *grid_map, LidarParams *lidar_param, cudaMat::SE3<float>& T_wc, pcl::PointCloud<pcl::PointXYZ>& lidar_points){
        Vector3f* point_values;
        size_t num_elements = lidar_param->vertical_lines * lidar_param->horizontal_num;
        cudaMallocManaged(&point_values, num_elements * sizeof(Vector3f));

        // 在GPU上启动核函数
        lidarRaycastKernel<<<lidar_param->vertical_lines, lidar_param->horizontal_num>>>(point_values, *grid_map, *lidar_param, T_wc);
        
        cudaDeviceSynchronize();

        std::vector<Vector3f> cpu_points(num_elements);
        cudaMemcpy(cpu_points.data(), point_values, num_elements * sizeof(Vector3f), cudaMemcpyDeviceToHost);
        
        lidar_points.points.clear();
        lidar_points.points.reserve(num_elements);
        
        for (const auto& point : cpu_points) {
            if (point.x != 0 || point.y != 0 || point.z != 0) {
                lidar_points.points.emplace_back(point.x, point.y, point.z);
            }
        }
        cudaFree(point_values);
        return;
    }


    DynamicDepthRenderer::DynamicDepthRenderer(const CameraParams &camera_param,
                                               int max_box_count)
        : camera_param_(camera_param), max_box_count_(max_box_count)
    {
        if (max_box_count_ <= 0)
            throw std::invalid_argument("max_box_count must be positive");

        const size_t depth_elements = static_cast<size_t>(camera_param_.image_width) *
                                      static_cast<size_t>(camera_param_.image_height);
        cudaMalloc(reinterpret_cast<void **>(&boxes_cuda_),
                   static_cast<size_t>(max_box_count_) * sizeof(DynamicBox));
        cudaMalloc(reinterpret_cast<void **>(&depth_cuda_),
                   depth_elements * sizeof(float));
    }

    DynamicDepthRenderer::~DynamicDepthRenderer()
    {
        if (boxes_cuda_ != nullptr)
            cudaFree(boxes_cuda_);
        if (depth_cuda_ != nullptr)
            cudaFree(depth_cuda_);
    }

    void DynamicDepthRenderer::render(const std::vector<DynamicBox> &boxes,
                                      const cudaMat::SE3<float> &T_wc,
                                      cv::Mat &depth_image)
    {
        if (static_cast<int>(boxes.size()) > max_box_count_)
            throw std::runtime_error("dynamic box count exceeds renderer capacity");

        if (!boxes.empty())
        {
            cudaMemcpy(boxes_cuda_, boxes.data(), boxes.size() * sizeof(DynamicBox),
                       cudaMemcpyHostToDevice);
        }

        dynamicCameraRaycastKernel<<<camera_param_.image_height, camera_param_.image_width>>>(
            depth_cuda_, boxes_cuda_, static_cast<int>(boxes.size()), camera_param_, T_wc);

        const cudaError_t kernel_error = cudaGetLastError();
        if (kernel_error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(kernel_error));
        const cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(sync_error));

        depth_image.create(camera_param_.image_height, camera_param_.image_width, CV_32FC1);
        const size_t depth_bytes = static_cast<size_t>(camera_param_.image_width) *
                                   static_cast<size_t>(camera_param_.image_height) * sizeof(float);
        cudaMemcpy(depth_image.data, depth_cuda_, depth_bytes, cudaMemcpyDeviceToHost);
    }

    
}