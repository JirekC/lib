#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include "fas.hpp"

using namespace fas;

void field::Prepare(bool want_rms) {
    calc_rms = want_rms;
    // create & allocate buffers, create command queue
    //std::cout << "Allocate memory on the device.\n";
    try {
        size_t elements = (size_t)size.x * size.y * size.z;
        buff_A = std::move(cl::Buffer( d->cl_context, CL_MEM_READ_WRITE, sizeof(data_t) * elements ));
        buff_B = std::move(cl::Buffer( d->cl_context, CL_MEM_READ_WRITE, sizeof(data_t) * elements ));
        if (calc_rms)
        {
            buff_rms = std::move(cl::Buffer( d->cl_context, CL_MEM_READ_WRITE, sizeof(data_t) * elements ));
        }
        buff_mat = std::move(cl::Buffer(d->cl_context, CL_MEM_READ_WRITE, sizeof(uint8_t) * elements ));
        buff_c = std::move(cl::Buffer(d->cl_context, CL_MEM_READ_ONLY, sizeof(data_t) * 256 )); // maximum 256 materials in one simulation
        buff_r = std::move(cl::Buffer( d->cl_context, CL_MEM_READ_ONLY, sizeof(data_t) * 256 ));
        // create command queue
        cl_queue = std::move(cl::CommandQueue(d->cl_context, *(d->phy_dev), CL_QUEUE_PROFILING_ENABLE));
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't allocate memory on the device or create command queue (fas::field::Prepare()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
    // copy materials to device
    try {
        data_t * c_arr = static_cast<data_t*>(cl_queue.enqueueMapBuffer(buff_c, CL_TRUE, CL_MAP_WRITE, 0, sizeof(data_t) * 256));
        data_t * r_arr = static_cast<data_t*>(cl_queue.enqueueMapBuffer(buff_r, CL_TRUE, CL_MAP_WRITE, 0, sizeof(data_t) * 256));
        if (materials.size() > 256)
            materials.resize(256); // cut out-of-range (max 256 materials can be used)
        for (int i = 0; i < (int)materials.size(); i++) {
            c_arr[i] = materials[i].c;
            r_arr[i] = materials[i].r;
        }
        cl_queue.enqueueUnmapMemObject(buff_r, r_arr);
        cl_queue.enqueueUnmapMemObject(buff_c, c_arr);
        cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't copy field.materials to device (fas::field::Prepare()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void field::Clear() {
    try {
        d->clear_kernel.setArg(0, buff_A);
        d->clear_kernel.setArg(1, buff_B);
        d->clear_kernel.setArg(2, buff_mat);
        if (calc_rms)
            d->clear_kernel.setArg(3, buff_rms);
        else
            d->clear_kernel.setArg(3, sizeof(cl_mem*), cl_mem(NULL));
        cl_queue.enqueueNDRangeKernel(d->clear_kernel, { 0,0,0 }, { size.x, size.y, size.z });
        //cl_queue.finish(); // wait for device
        cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't clear acoustic field (fas::field::Clear()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
    steps_calculated = 0; // reset counter
}

void field::SimStep() {
    try {
        // integrate actual state of field
        if (calc_rms) {
            data_t w;
            if (steps_calculated >= rms_window.size())
                w = 0.0;
            else
                w = rms_window[steps_calculated];
            d->rms_sum_kernel.setArg(0, p_buff ? buff_B : buff_A);
            d->rms_sum_kernel.setArg(1, buff_rms);
            d->rms_sum_kernel.setArg(2, w);
            cl_queue.enqueueNDRangeKernel(d->rms_sum_kernel, { 0,0,0 }, { size.x, size.y, size.z });
        }
        steps_calculated++; // update step counter - now, holds number of calculated steps
        // calculate next state ( p(t+1) )
        if (p_buff == 0) {
            d->sim_step_kernel.setArg(0, buff_A);
            d->sim_step_kernel.setArg(1, buff_B);
            p_buff = 1;
        }
        else {
            d->sim_step_kernel.setArg(0, buff_B);
            d->sim_step_kernel.setArg(1, buff_A);
            p_buff = 0;
        }
        d->sim_step_kernel.setArg(2, buff_mat);
        d->sim_step_kernel.setArg(3, buff_r);
        d->sim_step_kernel.setArg(4, buff_c);
        d->sim_step_kernel.setArg(5, size.z);
        d->sim_step_kernel.setArg(6, dx);
        d->sim_step_kernel.setArg(7, dt);
        cl::Event e;
        cl_queue.enqueueBarrierWithWaitList(); // wait for all previous work
        cl_queue.enqueueNDRangeKernel(d->sim_step_kernel, { 0,0 }, { size.x, size.y }, cl::NullRange, NULL, &e);
        // diagnostic only, comment if not used:
        Finish(); // global finish for all works
        uint64_t start, end;
        e.getProfilingInfo(CL_PROFILING_COMMAND_START, &start);
        e.getProfilingInfo(CL_PROFILING_COMMAND_END, &end);
        printf("   elapsed time [ns]: %lli", (end - start));
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't run simulation step (fas::field::SimStep()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void field::FinishRms() {
    try {
        if (!calc_rms)
            return;
        // normalized noise power gain
        data_t nnpg = 0;
        for (auto w : rms_window)
            nnpg += w * w;
        nnpg /= (data_t)steps_calculated;
        // 1/sqrt(nnpg) * sqrt(1/N * sum)
        d->rms_final_kernel.setArg(0, buff_rms);
        d->rms_final_kernel.setArg(1, (data_t)1.0 / (data_t)steps_calculated);
        d->rms_final_kernel.setArg(2, (data_t)1.0 / sqrt(nnpg));
        cl_queue.enqueueNDRangeKernel(d->rms_final_kernel, { 0,0,0 }, { size.x, size.y, size.z });
        cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't calculate rms value (fas::field::FinishRms()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

data_t * field::Map_p_t_read() {
    try {
        // first, unmap old mem region (if mapped)
        Unmap_p_t();
        // then map new
        if (p_buff == 0) {
            p_mapped_ptr = static_cast<fas::data_t*>(cl_queue.enqueueMapBuffer(buff_A, CL_TRUE, CL_MAP_READ, 0, sizeof(data_t) * size.x * size.y * size.z));
            if (p_mapped_ptr) {
                p_mapped_buff = &buff_A; // success
            }
        }
        else
        {
            p_mapped_ptr = static_cast<fas::data_t*>(cl_queue.enqueueMapBuffer(buff_B, CL_TRUE, CL_MAP_READ, 0, sizeof(data_t) * size.x * size.y * size.z));
            if (p_mapped_ptr) {
                p_mapped_buff = &buff_B;
            }
        }
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't map device memory for read (fas::field::Map_p_t_read()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    return p_mapped_ptr;
}

data_t * field::Map_p_t_write() {
    try {
        // first, unmap old mem region (if mapped)
        Unmap_p_t();
        // then map new
        if (p_buff == 0) {
            p_mapped_ptr = static_cast<fas::data_t*>(cl_queue.enqueueMapBuffer(buff_A, CL_TRUE, CL_MAP_WRITE, 0, sizeof(data_t) * size.x * size.y * size.z));
            if (p_mapped_ptr) {
                p_mapped_buff = &buff_A; // success
            }
        }
        else
        {
            p_mapped_ptr = static_cast<fas::data_t*>(cl_queue.enqueueMapBuffer(buff_B, CL_TRUE, CL_MAP_WRITE, 0, sizeof(data_t) * size.x * size.y * size.z));
            if (p_mapped_ptr) {
                p_mapped_buff = &buff_B;
            }
        }
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't map device memory for write (fas::field::Map_p_t_write()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    return p_mapped_ptr;
}

void field::Unmap_p_t() {
    if (p_mapped_ptr) {
        cl_queue.enqueueUnmapMemObject(*p_mapped_buff, p_mapped_ptr);
        cl_queue.enqueueBarrierWithWaitList(); // for write operation
    }
    p_mapped_ptr = nullptr;
    p_mapped_buff = nullptr;
}

data_t * field::Map_rms_read() {
    try {
        // first, unmap old mem region (if mapped)
        Unmap_rms();
        // then map new
        rms_mapped_ptr = (fas::data_t*)(cl_queue.enqueueMapBuffer(buff_rms, CL_TRUE, CL_MAP_READ, 0, sizeof(data_t) * size.x * size.y * size.z));
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't map device memory for read (fas::field::Map_rms_read()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    return rms_mapped_ptr;
}

void field::Unmap_rms() {
    if (rms_mapped_ptr)
        cl_queue.enqueueUnmapMemObject(buff_rms, rms_mapped_ptr);
    rms_mapped_ptr = nullptr;
}
