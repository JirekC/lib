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
