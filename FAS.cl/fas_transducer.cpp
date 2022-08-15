#include "fas.hpp"
#include "fas_math.hpp"

using namespace fas;

void transducer::CollectElements() {
    
    // TODO: try-catch, check algorithm & comment well

    cl::Buffer buff_count(f->d->cl_context, CL_MEM_READ_WRITE, sizeof(uint32_t) * f->size.y * f->size.z);
    cl::Buffer buff_psum(f->d->cl_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * f->size.y * f->size.z);
    cl::Buffer buff_tmp_last_col( f->d->cl_context, CL_MEM_READ_WRITE, sizeof(uint64_t) * f->size.z );

    // count number of transducer's elements (with MSB set) in each line along the x-axis
    f->d->tdcr_count_elements_kernel.setArg(0, f->buff_mat);
    f->d->tdcr_count_elements_kernel.setArg(1, buff_count);
    f->d->tdcr_count_elements_kernel.setArg(2, f->size.x);
    f->cl_queue.enqueueNDRangeKernel(f->d->tdcr_count_elements_kernel, { 0,0 }, { f->size.y, f->size.z });
    f->cl_queue.enqueueBarrierWithWaitList();

    // for kernel debug only
//    uint32_t* count_arr = static_cast<uint32_t*>(f->cl_queue.enqueueMapBuffer(buff_count, CL_TRUE, CL_MAP_READ, 0, sizeof(uint32_t) * f->size.y * f->size.z));
//    f->cl_queue.enqueueUnmapMemObject(buff_count, count_arr);
//    f->cl_queue.enqueueBarrierWithWaitList();

    // calc prefix sum of 2D (count array)
    f->d->horizontal_prefix_sum_uint_ulong_kernel.setArg(0, buff_count);
    f->d->horizontal_prefix_sum_uint_ulong_kernel.setArg(1, buff_psum);
    f->d->horizontal_prefix_sum_uint_ulong_kernel.setArg(2, buff_tmp_last_col);
    f->d->horizontal_prefix_sum_uint_ulong_kernel.setArg(3, f->size.y);
    f->cl_queue.enqueueNDRangeKernel(f->d->horizontal_prefix_sum_uint_ulong_kernel, 0, f->size.z);
    f->cl_queue.enqueueBarrierWithWaitList();

    // for kernel debug only
//    uint64_t* psum_arr = static_cast<uint64_t*>(f->cl_queue.enqueueMapBuffer(buff_psum, CL_TRUE, CL_MAP_READ, 0, sizeof(uint64_t) * f->size.y * f->size.z));
    // not for debug, must be leaved as is ->>
    // wait here to finish previous kernel and then map memory (enqueueBarrier & blocking = true)
    uint64_t* tmp_last_col_arr = static_cast<uint64_t*>(f->cl_queue.enqueueMapBuffer(buff_tmp_last_col, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(uint64_t) * f->size.z));
    for (uint32_t i = 1; i < f->size.z; i++) {
        tmp_last_col_arr[i] += tmp_last_col_arr[i - 1];
    }
    f->cl_queue.enqueueUnmapMemObject(buff_tmp_last_col, tmp_last_col_arr);
    // <<-
//    f->cl_queue.enqueueUnmapMemObject(buff_psum, psum_arr);
    f->cl_queue.enqueueBarrierWithWaitList();

    f->d->vertical_prexix_sum_ulong_kernel.setArg(0, buff_psum);
    f->d->vertical_prexix_sum_ulong_kernel.setArg(1, buff_tmp_last_col);
    f->d->vertical_prexix_sum_ulong_kernel.setArg(2, f->size.z);
    f->cl_queue.enqueueNDRangeKernel(f->d->vertical_prexix_sum_ulong_kernel, 0, f->size.y);
    f->cl_queue.enqueueBarrierWithWaitList();

    // for kernel debug only
//    psum_arr = static_cast<uint64_t*>(f->cl_queue.enqueueMapBuffer(buff_psum, CL_TRUE, CL_MAP_READ, 0, sizeof(uint64_t) * f->size.y * f->size.z));
//    f->cl_queue.enqueueUnmapMemObject(buff_psum, psum_arr);
//    f->cl_queue.enqueueBarrierWithWaitList();

    // collect element's coordinates into buff_elements array
    // read total number of elements = buff_tmp_last_col[last]
    tmp_last_col_arr = static_cast<uint64_t*>(f->cl_queue.enqueueMapBuffer(buff_tmp_last_col, CL_TRUE, CL_MAP_READ, 0, sizeof(uint64_t) * f->size.z));
    num_elements = tmp_last_col_arr[f->size.z - 1]; // TODO: map only last element
    f->cl_queue.enqueueUnmapMemObject(buff_tmp_last_col, tmp_last_col_arr);
    f->cl_queue.enqueueBarrierWithWaitList();
    // allocate memory on device
    buff_elements = std::move(cl::Buffer( f->d->cl_context, CL_MEM_READ_WRITE, sizeof(uint32_t) * 3 * num_elements ));
    f->d->tdcr_collect_elements_kernel.setArg(0, f->buff_mat);
    f->d->tdcr_collect_elements_kernel.setArg(1, buff_psum);
    f->d->tdcr_collect_elements_kernel.setArg(2, buff_elements);
    f->d->tdcr_collect_elements_kernel.setArg(3, f->size.x);
    f->d->tdcr_collect_elements_kernel.setArg(4, static_cast<uint64_t>(num_elements));
    f->cl_queue.enqueueNDRangeKernel(f->d->tdcr_collect_elements_kernel, { 0,0 }, { f->size.y,f->size.z });
    f->cl_queue.enqueueBarrierWithWaitList();
    // now all element's coordinates are stored in buff_elements
    // clear all MSBs of field.material
    f->d->tdcr_clear_mat_MSBs_kernel.setArg(0, f->buff_mat);
    f->cl_queue.enqueueNDRangeKernel(f->d->tdcr_clear_mat_MSBs_kernel, { 0,0,0 }, { f->size.x,f->size.y,f->size.z });
    f->cl_queue.finish();
}

std::vector<uint32_t> transducer::GetElementsCoords() {
    std::vector<uint32_t> rv;
    rv.resize(num_elements * 3);
    void* hptr;
    try {
        hptr = f->cl_queue.enqueueMapBuffer(buff_elements, CL_TRUE, CL_MAP_READ, 0, sizeof(uint32_t) * 3 * num_elements);
        memcpy(rv.data(), hptr, sizeof(uint32_t) * 3 * num_elements);
        f->cl_queue.enqueueUnmapMemObject(buff_elements, hptr);
        f->cl_queue.enqueueBarrierWithWaitList();
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't map device buffer. (fas::transducer::GetElementsCoords()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }

    return std::move(rv);
}

void driver::Drive(data_t signal) {
    try {
        if (f->p_buff == 0) {
            f->d->drive_kernel.setArg(0, f->buff_A);
        }
        else {
            f->d->drive_kernel.setArg(0, f->buff_B);
        }
        f->d->drive_kernel.setArg(1, signal);
        f->d->drive_kernel.setArg(2, buff_elements);
        f->d->drive_kernel.setArg(3, f->size.x);
        f->d->drive_kernel.setArg(4, f->size.y);

        cl::Event evt;
        f->cl_queue.enqueueNDRangeKernel(f->d->drive_kernel, 0, num_elements, cl::NullRange, NULL, &evt);
        //f->cl_queue.finish(); // wait for device
        /*evt.wait();
        static int64_t duration_ns = 0;
        duration_ns += evt.getProfilingInfo<CL_PROFILING_COMMAND_END>() - evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();*/
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't drive acoustic pressure - set field.buff_A/B (fas::driver::Drive()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

// void scanner::CollectElements() {
//     transducer::CollectElements();
//     if (data != nullptr) delete[] data;
//     // alloc memory for scanned data (host)
//     data = new data_t[num_elements];
//     // alloc memory for scanned data (device)
//     try {
//         buff_data = std::move(cl::Buffer(f->d->cl_context, CL_MEM_READ_WRITE, sizeof(data_t) * num_elements));
//     }
//     catch (cl::Error& e) {
//         // "data" will be deallocated in this->destructor
//         std::string s;
//         s = "ERR: Can't allocate memory on device (fas::scanner::CollectElements()):\n";
//         s += e.what();
//         throw std::runtime_error(s);
//     }
// }

void scanner::Prepare(field &_f, vec3<uint32_t> position, vec3<double> rotation, vec2<uint32_t> size, std::string out_file_name, uint32_t _store_every_nth_frame)
{
    f = &_f;
    num_elements = (size_t)(size.x) * size.y;
    store_every_nth_frame = _store_every_nth_frame;

    try {
        // allocate memory for scanned data on device
        buff_data = std::move(cl::Buffer(f->d->cl_context, CL_MEM_READ_WRITE/*CL_MEM_WRITE_ONLY*/, sizeof(data_t) * num_elements));
        buff_elements = std::move(cl::Buffer(f->d->cl_context, CL_MEM_READ_WRITE/*CL_MEM_READ_ONLY*/, sizeof(uint32_t) * num_elements * 3));

        //calc & store coordinates
        uint32_t *tmp_elements = static_cast<uint32_t*>(f->cl_queue.enqueueMapBuffer(buff_elements, CL_TRUE, CL_MAP_WRITE, 0, sizeof(uint32_t) * 3 * num_elements));
        mat3_3 rot = RotationMatrix<data_t>(rotation);
        for(uint32_t y = 0; y < size.y; y++)
        {
            for(uint32_t x = 0; x < size.x; x++)
            {
                // rotate & translate, note: "z" is alway 0 so not used
                float new_xf = rot.a11 * (data_t)x + rot.a12 * (data_t)y + (data_t)(position.x);
                float new_yf = rot.a21 * (data_t)x + rot.a22 * (data_t)y + (data_t)(position.y);
                float new_zf = rot.a31 * (data_t)x + rot.a32 * (data_t)y + (data_t)(position.z);
                // convert to integer and check for field boundary
                uint32_t new_x = (new_xf < 0.0f) ? 0 : new_xf;
                uint32_t new_y = (new_yf < 0.0f) ? 0 : new_yf;
                uint32_t new_z = (new_zf < 0.0f) ? 0 : new_zf;
                new_x = (new_x >= f->size.x) ? f->size.x - 1 : new_x;
                new_y = (new_y >= f->size.y) ? f->size.y - 1 : new_y;
                new_z = (new_z >= f->size.z) ? f->size.z - 1 : new_z;
                // store
                tmp_elements[(size_t)y * size.x + x] = new_x;
                tmp_elements[(size_t)y * size.x + x + num_elements] = new_y;
                tmp_elements[(size_t)y * size.x + x + 2 * num_elements] = new_z;
            }
        }
        f->cl_queue.enqueueUnmapMemObject(buff_elements, tmp_elements);
    }
    catch (cl::Error& e) {
        throw std::runtime_error("ERR: Can't allocate memory on device (fas::scanner::Prepare()):\n" + std::string(e.what()));
    }
    // output file
    try {
        out_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        out_file.open(out_file_name, std::ios::binary | std::ios::trunc); // sdt::exception will go higher, if thrown
    }
    catch (std::exception& e)
    {
        throw std::runtime_error("ERR: Can't write to file \"" + out_file_name + "\" (fas::scanner::Prepare()):\n" + std::string(e.what()));
    }

}

void scanner::Scan2devmem()
{
    if(f->steps_calculated % store_every_nth_frame != 0)
    {
        return; // nothing to do - don't store this frame
    }
    try {
        if (f->p_buff == 0)
            f->d->scan_kernel.setArg(0, f->buff_A);
        else
            f->d->scan_kernel.setArg(0, f->buff_B);
        f->d->scan_kernel.setArg(1, buff_elements);
        f->d->scan_kernel.setArg(2, buff_data);
        f->d->scan_kernel.setArg(3, f->size.x);
        f->d->scan_kernel.setArg(4, f->size.y);
        f->cl_queue.enqueueNDRangeKernel(f->d->scan_kernel, 0, num_elements);
    }
    catch (cl::Error& e) {
        std::string s;
        s = "ERR: Can't scan acoustic pressure (fas::scanner::Scan2devmem()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}

void scanner::Scan2file() {
    if(f->steps_calculated % store_every_nth_frame != 0)
    {
        return; // nothing to do - don't store this frame
    }
    void* hptr = nullptr;
    try {
        hptr = f->cl_queue.enqueueMapBuffer(buff_data, CL_TRUE, CL_MAP_READ, 0, sizeof(data_t) * num_elements);
        out_file.write((char*)hptr, sizeof(data_t) * num_elements);
        f->cl_queue.enqueueUnmapMemObject(buff_data, hptr);
        f->cl_queue.enqueueBarrierWithWaitList(); // for unmap ( ?? )
    }
    catch (cl::Error& e) {
        f->cl_queue.enqueueUnmapMemObject(buff_data, hptr); // in case of file IO error
        std::string s;
        s = "ERR: Can't read data from device to file. (fas::scanner::Scan2file()):\n";
        s += e.what();
        throw std::runtime_error(s);
    }
}
