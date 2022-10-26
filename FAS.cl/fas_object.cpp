#include <fstream>
#include <iostream>
#include "fas.hpp"
#include "fas_math.hpp"

using namespace fas;

void object::CreateRect(field& f, vec3<uint32_t> pos, vec3<double> rot, vec2<uint32_t> size, uint8_t material) {
    mat3_3<data_t> rmatv = RotationMatrix<data_t>(rot);
    data_t rot_pos[] = {   rmatv.a11, rmatv.a12, rmatv.a13, rmatv.a21, rmatv.a22, rmatv.a23, rmatv.a31, rmatv.a32, rmatv.a33,
                        pos.x, pos.y, pos.z };

    try {
        // alloc buffer for rotation matrix in device's global mem & copy data - will be deallocated by destructor
        cl::Buffer buff_rot_pos = cl::Buffer(f.d->cl_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 12, rot_pos);

        f.d->object_rect_kernel.setArg(0, f.buff_mat);
        f.d->object_rect_kernel.setArg(1, buff_rot_pos);
        f.d->object_rect_kernel.setArg(2, f.size.x);
        f.d->object_rect_kernel.setArg(3, f.size.y);
        f.d->object_rect_kernel.setArg(4, f.size.z);
        f.d->object_rect_kernel.setArg(5, material);
        f.cl_queue.enqueueNDRangeKernel(f.d->object_rect_kernel, { 0,0 }, { (size_t)size.x * 2,(size_t)size.y * 2 });
        f.cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create 3D object - set field.buff_mat (fas::object::CreateRect):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void object::CreateEllipse(field& f, vec3<uint32_t> pos, vec3<double> rot, vec2<uint32_t> size, uint8_t material) {
    mat3_3<data_t> rmatv = RotationMatrix<data_t>(rot);
    data_t rot_pos[] = { rmatv.a11, rmatv.a12, rmatv.a13, rmatv.a21, rmatv.a22, rmatv.a23, rmatv.a31, rmatv.a32, rmatv.a33,
                        pos.x, pos.y, pos.z };

    try {
        // alloc buffer for rotation matrix in device's global mem & copy data - will be deallocated by destructor
        cl::Buffer buff_rot_pos = cl::Buffer(f.d->cl_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 12, rot_pos);

        f.d->object_ellipse_kernel.setArg(0, f.buff_mat);
        f.d->object_ellipse_kernel.setArg(1, buff_rot_pos);
        f.d->object_ellipse_kernel.setArg(2, f.size.x);
        f.d->object_ellipse_kernel.setArg(3, f.size.y);
        f.d->object_ellipse_kernel.setArg(4, f.size.z);
        f.d->object_ellipse_kernel.setArg(5, material);
        f.cl_queue.enqueueNDRangeKernel(f.d->object_ellipse_kernel, { 0,0 }, { (size_t)size.x * 4,(size_t)size.y * 4 });
        f.cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create 3D object - set field.buff_mat (fas::object::CreateEllipse):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void object::CreateBox(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material) {
    mat3_3<data_t> rmatv = RotationMatrix<data_t>(rot);
    data_t rot_pos[] = { rmatv.a11, rmatv.a12, rmatv.a13, rmatv.a21, rmatv.a22, rmatv.a23, rmatv.a31, rmatv.a32, rmatv.a33,
                        pos.x, pos.y, pos.z };

    try {
        // alloc buffer for rotation matrix in device's global mem & copy data - will be deallocated by destructor
        cl::Buffer buff_rot_pos = cl::Buffer(f.d->cl_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 12, rot_pos);

        f.d->object_box_kernel.setArg(0, f.buff_mat);
        f.d->object_box_kernel.setArg(1, buff_rot_pos);
        f.d->object_box_kernel.setArg(2, f.size.x);
        f.d->object_box_kernel.setArg(3, f.size.y);
        f.d->object_box_kernel.setArg(4, f.size.z);
        f.d->object_box_kernel.setArg(5, material);
        f.cl_queue.enqueueNDRangeKernel(f.d->object_box_kernel, { 0,0,0 }, { (size_t)size.x * 2,(size_t)size.y * 2,(size_t)size.z * 2 });
        f.cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create 3D object - set field.buff_mat (fas::object::CreateBox):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void object::CreateCylinder(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material) {
    mat3_3<data_t> rmatv = RotationMatrix<data_t>(rot);
    data_t rot_pos[] = { rmatv.a11, rmatv.a12, rmatv.a13, rmatv.a21, rmatv.a22, rmatv.a23, rmatv.a31, rmatv.a32, rmatv.a33,
                        pos.x, pos.y, pos.z };

    try {
        // alloc buffer for rotation matrix in device's global mem & copy data - will be deallocated by destructor
        cl::Buffer buff_rot_pos = cl::Buffer(f.d->cl_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 12, rot_pos);

        f.d->object_cylinder_kernel.setArg(0, f.buff_mat);
        f.d->object_cylinder_kernel.setArg(1, buff_rot_pos);
        f.d->object_cylinder_kernel.setArg(2, f.size.x);
        f.d->object_cylinder_kernel.setArg(3, f.size.y);
        f.d->object_cylinder_kernel.setArg(4, f.size.z);
        f.d->object_cylinder_kernel.setArg(5, material);
        f.d->object_cylinder_kernel.setArg(6, size.z * 2);
        f.cl_queue.enqueueNDRangeKernel(f.d->object_cylinder_kernel, { 0,0 }, { (size_t)size.x * 4,(size_t)size.y * 4 });
        f.cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create 3D object - set field.buff_mat (fas::object::CreateCylinder):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void object::CreateEllipsoid(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material) {
    mat3_3<data_t> rmatv = RotationMatrix<data_t>(rot);
    data_t rot_pos[] = { rmatv.a11, rmatv.a12, rmatv.a13, rmatv.a21, rmatv.a22, rmatv.a23, rmatv.a31, rmatv.a32, rmatv.a33,
                        pos.x, pos.y, pos.z };

    try {
        // alloc buffer for rotation matrix in device's global mem & copy data - will be deallocated by destructor
        cl::Buffer buff_rot_pos = cl::Buffer(f.d->cl_context, CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY, sizeof(data_t) * 12, rot_pos);

        f.d->object_ellipsoid_kernel.setArg(0, f.buff_mat);
        f.d->object_ellipsoid_kernel.setArg(1, buff_rot_pos);
        f.d->object_ellipsoid_kernel.setArg(2, f.size.x);
        f.d->object_ellipsoid_kernel.setArg(3, f.size.y);
        f.d->object_ellipsoid_kernel.setArg(4, f.size.z);
        f.d->object_ellipsoid_kernel.setArg(5, material);
        f.cl_queue.enqueueNDRangeKernel(f.d->object_ellipsoid_kernel, { 0,0,0 }, { (size_t)size.x * 4,(size_t)size.y * 4,(size_t)size.z * 4 });
        f.cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create 3D object - set field.buff_mat (fas::object::CreateEllipsoid):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void object::LoadVoxelMap(field& f, const char *path){
    uint8_t *p_mapped_ptr;

    try {
        p_mapped_ptr = static_cast<uint8_t*>(f.cl_queue.enqueueMapBuffer(f.buff_mat, CL_TRUE, CL_MAP_WRITE, 0, 
                                            sizeof(uint8_t) * f.size.x * f.size.y * f.size.z));
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't map device memory for write (fas::object::LoadVoxelMap()):\n";
        s += e.what();
        s += std::to_string(e.err());
        throw std::runtime_error(s);
    }

    try {
        std::ifstream vox_file;
        vox_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
		vox_file.open(path, std::ios_base::binary);
        std::vector<uint8_t> slice;
		slice.reserve((size_t)f.size.x * f.size.y);
        for(uint32_t z = 0; z < f.size.z; z++) {
            // read one slice (x, y = var.; z = const)
			vox_file.read((char*)slice.data(), (size_t)f.size.x * f.size.y);
			// find non-zero voxels
			size_t idx = 0;
			for(uint32_t y = 0; y < f.size.y; y++)
			{
				for(uint32_t x = 0; x < f.size.x; x++)
				{
					uint8_t material = slice[idx++];
					if(material != 0)
					{
						// store non-zero voxel's to GPU
                        p_mapped_ptr[z*f.size.x*f.size.y + y*f.size.x + x] = material;
					}
				}
			}
        }
    }
    catch (std::exception& e) {
        // f.cl_queue.enqueueUnmapMemObject(f.buff_mat, p_mapped_ptr);
        // f.cl_queue.enqueueBarrierWithWaitList(); // for write operation
        std::string s;
        s = "ERR: Can't load voxel map from file \"" + std::string(path) + "\" (fas::object::LoadVoxelMap()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    // unmap material buffer
    f.cl_queue.enqueueUnmapMemObject(f.buff_mat, p_mapped_ptr);
    f.cl_queue.enqueueBarrierWithWaitList(); // for write operation
}
