#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include "fas.hpp"

using namespace fas; // because I'm lazy ...

void device::Prepare(cl::Device& dev, const char* ocl_path) {

    // create context on selected platform / device
    try {
        cl_context = std::move(cl::Context({ dev }));
        phy_dev = &dev;
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create OpenCL context (fas::device::Prepare()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    // load & build program
    std::cout << "Load and build OpenCL program for " << dev.getInfo<CL_DEVICE_NAME>() << "\n";
    std::ifstream cl_src_file(ocl_path, std::ios::in);
    if (cl_src_file.is_open() == false) {
        std::string s = "ERR: Can't open OpenCL source file: \"";
        s += ocl_path;
        s += "\" (fas::device::Prepare())";
        throw std::runtime_error(s);
    }
    std::stringstream strStream;
    strStream << cl_src_file.rdbuf();
    try {
        cl_program = std::move(cl::Program(cl_context, strStream.str(), false));
        cl_program.build("");// -cl - no - signed - zeros - cl - fast - relaxed - math");
    }
    catch (cl::Error& e)
    {
        if (e.err() == CL_BUILD_PROGRAM_FAILURE)
        {
            std::string s;
            s = "ERR: OpenCL program build failed (fas::device::Prepare()):\n" + cl_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev);
            throw std::runtime_error(s);
        }
        else {
            std::string s;
            s = "ERR: OpenCL program error (fas::device::Prepare()):\n";
            s += e.what();
            throw std::runtime_error(s);
        }
    }

    // create kernels
    std::cout << "Creating kernels\n";
    try {
        clear_kernel = std::move(cl::Kernel( cl_program, "clear" ));
        sim_step_kernel = std::move(cl::Kernel( cl_program, "sim_step" ));
        rms_sum_kernel = std::move(cl::Kernel( cl_program, "rms_sum" ));
        rms_final_kernel = std::move(cl::Kernel( cl_program, "rms_final" ));
        object_rect_kernel = std::move(cl::Kernel(cl_program, "object_rect"));
        object_ellipse_kernel = std::move(cl::Kernel(cl_program, "object_ellipse"));
        object_box_kernel = std::move(cl::Kernel(cl_program, "object_box"));
        object_cylinder_kernel = std::move(cl::Kernel(cl_program, "object_cylinder"));
        object_ellipsoid_kernel = std::move(cl::Kernel( cl_program, "object_ellipsoid" ));
        tdcr_count_elements_kernel = std::move(cl::Kernel( cl_program, "tdcr_count_elements" ));
        tdcr_collect_elements_kernel = std::move(cl::Kernel( cl_program, "tdcr_collect_elements" ));
        tdcr_clear_mat_MSBs_kernel = std::move(cl::Kernel( cl_program, "tdcr_clear_mat_MSBs" ));
        drive_kernel = std::move(cl::Kernel(cl_program, "drive"));
        scan_kernel = std::move(cl::Kernel( cl_program, "scan" ));
        horizontal_prefix_sum_uint_ulong_kernel = std::move(cl::Kernel( cl_program, "horizontal_prefix_sum_uint_ulong" ));
        vertical_prexix_sum_ulong_kernel = std::move(cl::Kernel( cl_program, "vertical_prexix_sum_ulong" ));
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't create kernels (fas::device::Prepare()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
    
  //  std::cout << "CL_KERNEL_WORK_GROUP_SIZE: " << sim_step_kernel.getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(*phy_dev) << std::endl;
  //  std::cout << "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE: " << sim_step_kernel.getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(*phy_dev) << std::endl;
  //  std::cout << "CL_KERNEL_PRIVATE_MEM_SIZE: " << sim_step_kernel.getWorkGroupInfo<CL_KERNEL_PRIVATE_MEM_SIZE>(*phy_dev) << std::endl;
  //  std::cout << "CL_KERNEL_COMPILE_WORK_GROUP_SIZE: " << sim_step_kernel.getWorkGroupInfo<CL_KERNEL_COMPILE_WORK_GROUP_SIZE>(*phy_dev) << std::endl;
}
