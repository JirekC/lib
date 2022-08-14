#ifndef FAS_DATA_TYPES_H
#define FAS_DATA_TYPES_H

/** 
 * @file 
 * @brief Data types / structures definition; some common macros definition
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <stdint.h>

#define INDEX3D(x, y, z) ((size_t)(z) * (size_t)x_size * (size_t)y_size + (size_t)(y) * (size_t)x_size + (size_t)(x))
#define INDEX2D(u, v) ((size_t)(v) * (size_t)u_size + (size_t)(u))

#ifndef M_PI
	#define M_PI 3.14159265358979323846264338327950288
#endif /* M_PI */

namespace fas {

	// data types
	typedef float data_t; ///< type of numerical data, change OpenCL kernels if changed to double

	/** \brief 2D vector */
	template<typename T = data_t>
	struct vec2 {
		T x, y;
	};

	/** \brief 3D vector */
	template<typename T = data_t>
	struct vec3 {
		T x, y, z;
	};

	/** \brief 3x3 matrix */
	template<typename T = data_t>
	struct mat3_3 {
		T a11, a12, a13;
		T a21, a22, a23;
		T a31, a32, a33;
	};

	/** \brief 4x4 matrix */
	template<typename T = data_t>
	struct mat4_4 {
		T a11, a12, a13, a14;
		T a21, a22, a23, a24;
		T a31, a32, a33, a34;
		T a41, a42, a43, a44;
	};

	/** \brief Acoustic enviroment - material definition
	*
	*	initialize \ref c an \ref ro for each material used and call \ref Recalc() before Preparing field
	**/
	struct material {
		data_t c; 			///< speed of sound [m * s^-1]
		data_t ro; 			///< density [kg * m^-3]
		data_t r; 			///< characteristic acoustic impedance (ro * c) calculated here for speed-up GPU calculations
		std::string name; 	///< human-readable name of material like "Air" ...

		/** \brief Recalculates materials from \b c and \b ro - must be initialized by user before call
		*   
		*	\param matvec vector of materials with \b c and \b ro initialized to proper value
		*	\param dx edge of cubic element [m]
		*/
		static void Recalc(std::vector<material>& matvec) {
			for( auto &ma : matvec ) {
				data_t c = ma.c;
				data_t ro = ma.ro;
				if (c <= 0.0 || ro <= 0.0) {
					std::string s;
					s = "fas::material::Recalc(): Material \"";
					s += ma.name;
					s += "\": c and ro properties must be greater than zero, but are ";
					s += std::to_string(c);
					s += " and ";
					s += std::to_string(ro);
					throw std::runtime_error(s);
				}
				ma.r = ro * c;
			}
		}

		/** \brief Recalculates f.materials
		*
		*	overload of function above, see documentation of it
		*	\param f field where vector<material> will be recalculated
		**/
		template<typename T = struct field>
		static void Recalc(T& f) {
			Recalc(f.materials);
		}
	};

	/** \brief specialized OpenCL device with compiled program and loaded kernels
	* 
	* Call \ref Prepare() first, before using this, or use constructor with \b device parameter
	**/
	struct device {
		cl::Device* phy_dev; ///< pointer to physical device (GPU for example)
		cl::Context cl_context;
		cl::Program cl_program;

		// CL simulation kernels
		cl::Kernel clear_kernel;
		cl::Kernel sim_step_kernel;
		cl::Kernel rms_sum_kernel;
		cl::Kernel rms_final_kernel;
		cl::Kernel object_rect_kernel;
		cl::Kernel object_ellipse_kernel;
		cl::Kernel object_box_kernel;
		cl::Kernel object_cylinder_kernel;
		cl::Kernel object_ellipsoid_kernel;
		cl::Kernel tdcr_count_elements_kernel;
		cl::Kernel tdcr_collect_elements_kernel;
		cl::Kernel tdcr_clear_mat_MSBs_kernel;
		cl::Kernel drive_kernel;
		cl::Kernel scan_kernel;
		cl::Kernel horizontal_prefix_sum_uint_ulong_kernel;
		cl::Kernel vertical_prexix_sum_ulong_kernel;

		device() { phy_dev = nullptr; }

		/** \brief Constructs, load program and compile it, init kernels
		* 
		*	\param dev one physical OpenCL device returned by \b cl::Platform::getDevices()
		*	\param ocl_path path (absolute or relative) to CL source code
		**/
		device(cl::Device& dev, const char* ocl_path = "fas.cl") { Prepare(dev, ocl_path); }

		/** \brief Load program and compile it, init kernels
		* 
		*	call it ONLY if constructor without parameters used, call it only ONCE
		*
		*	\param dev one physical OpenCL device returned by \b cl::Platform::getDevices()
		*	\param ocl_path path (absolute or relative) to CL source code
		**/
		void Prepare(cl::Device& dev, const char* ocl_path);
	};

	/** \brief 3D acoustic field of cubic elements.
	*
	* Call \ref Prepare() first, before using this
	**/
	struct field {
		device* d = nullptr; ///< OpenCl device on which field resides (and is calculated)
		cl::CommandQueue cl_queue; ///< OpenCL command queue on device, exclusive for each field
		vec3<uint32_t> size = { 3,3,3 }; ///< size of field [elements]
		data_t dx = 1e-6; ///< length of edge of CUBIC element [m]
		data_t dt = 1e-9; ///< simulation time step [s]
		size_t steps_calculated = 0; ///< number of calculated steps (duplicity here - holding this value for each field)

		// CL buffers
		cl::Buffer buff_A; ///< see \ref p_buff for explanation
		cl::Buffer buff_B; ///< see \ref p_buff for explanation
		cl::Buffer buff_rms; ///< buffer used for integration of RMS value
		cl::Buffer buff_mat; ///< material index used for each element
		cl::Buffer buff_r; ///< used materials (max 256), characteristic acoustic impedance
		cl::Buffer buff_c; ///< used materials (max 256), speed of sound
		int p_buff = 0; ///< 0: \b buff_A holds p(t) and \b buff_B holds p(t-1); 1: \b buff_B holds p(t) and \b buff_A holds p(t-1)
		data_t * p_mapped_ptr = nullptr;
		cl::Buffer * p_mapped_buff = nullptr;
		data_t * rms_mapped_ptr = nullptr;
		bool calc_rms = false; ///< true: there is requirement for calculating RMS value; false: no RMS calculation, \ref buff_rms not allocated

		std::vector<material> materials; ///< used materials, need to be initialised (calculated \b k & \b m ) before simulation, see \ref material::Recalc()
		std::vector<data_t> rms_window; ///< window used for "integrating": rect, Hann, Hamming ... One element of vector coresponding to one step of simulation
		//uint8_t symmetry = 0; ///< 0 by default (asymmetric field) else: see \ref FasFieldSetSymmetry() & \ref FAS_DIRECTION_ (please use only directions with "p" suffix)

		field() {}
		field(device& d) : d(&d) {} ///< \param d valid and initialized \ref device

		/**
		*	\param d valid and initialized \ref device
		*	\param size size of field in X, Y and Z coordinates (cartesian) [elements]
		*	\param dx length of cubic-element-edge [m]
		*	\param dt time step of simulation [s]
		**/
		field(device& d, vec3<uint32_t> size, data_t dx = 1e-3, data_t dt = 1e-6) : d(&d) {
			this->size.x = size.x < 3 ? 3 : size.x;
			this->size.y = size.y < 3 ? 3 : size.y;
			this->size.z = size.z < 3 ? 3 : size.z;
			this->dx = dx < 1e-6 ? 1e-6 : dx; // minimal element-edge is 1 um
			this->dt = dt < 1e-9 ? 1e-9 : dt; // minimal time step is 1 ns
		}

		void Prepare(bool want_rms = false); ///< Call only once, before use! \ref d must point to valid and initialized \ref device; \ref materials must be initialized, see \ref material::Recalc() \param want_rms set true if you want to calculate rms value in each element of field (need to define \ref rms_window before simulation)
		void Clear(); ///< Reset of simulation; for each element: sets pressure to 0.0, rms integration buffer to 0.0 and material to #0
		void SimStep(); ///< Run one step of simulation and swap pressure buffers (hiden to user)
		void FinishRms(); ///< Finishes calculation of RMS value in each element of field
		data_t * Map_p_t_read(); ///< Maps actual buffer (whole 3D array) with p(t) to host memory as read-only memory and return pointer to this new region. Region is valid untill new simulation step or field destruction.
		data_t * Map_p_t_write(); ///< Maps actual buffer (whole 3D array) with p(t) to host memory as write-only memory and return pointer to this new region. Region is valid untill new simulation step or field destruction. After finish of write call \ref Unmap_p_t() to update device's memory.
		void Unmap_p_t(); ///< Unmaps device memory from host mem space. Call it for update of devices memory after write. It is called implicitly from new Map_p_t_x() or destructor
		data_t * Map_rms_read(); ///< Maps rms buffer (whole 3D array) to host memory as read-only memory and return pointer to this new region. Region is valid untill new simulation step or field destruction.
		void Unmap_rms(); ///< Unmaps device memory from host mem space. It is called implicitly from new Map_rms_read() or destructor
		void Finish() { cl_queue.finish(); } ///< global finish for all works in command queue (unique for this field), blocking

		~field() {
			Unmap_p_t(); // not realy needed ( ? )
			Unmap_rms(); // not realy needed ( ? )
		}
	};

	/** \brief Static functions for "drawing" (set of material property of elements) objects in acoustic field */
	struct object {
		static void CreateRect(field& f, vec3<uint32_t> pos, vec3<double> rot, vec2<uint32_t> size, uint8_t material);
		static void CreateEllipse(field& f, vec3<uint32_t> pos, vec3<double> rot, vec2<uint32_t> size, uint8_t material);
		static void CreateBox(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material);
		static void CreateCylinder(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material); // size.y = base.radius_a, size.y = base.radius_b, size.z = height
		static void CreateEllipsoid(field& f, vec3<uint32_t> pos, vec3<double> rot, vec3<uint32_t> size, uint8_t material);
	};

	/** \brief Parent struct of driver and scanner (pressure "microphone" array) - colection of elements */
	struct transducer {
		field* f = nullptr; ///< pointer to acoustic field, where transducer exists
		size_t num_elements = 0; ///< number of elements of transducer
		cl::Buffer buff_elements; ///< coordinate of each element of transducer format: uint32 { x0, x1 ... xn, y1 .. yn, z1 .. zn }

		transducer() {}
		/** \param f acoustic field, where transducer will exists **/
		transducer(field& f) {
			this->f = &f;
		}

		void CollectElements(); ///< Store coordinates of transducer's elements - elements with MSB of material-attribute set; use \ref fas::object with \b material 0x80 and above to create driver shape and than call this fnc
		std::vector<uint32_t> GetElementsCoords(); ///< Return vector of coordinates of transducer's elements, format: uint32 { x0, x1 ... xn, y1 .. yn, z1 .. zn }
	};

	/** \brief Driver - sets/drive acoustic pressure in each element */
	struct driver : transducer {
		driver() {}
		driver(field& f) : transducer(f) {}

		/** \brief Sets pressure in each element of transducer to value of \b signal
		 *
		 * Coordinates of elements must be already initialized by \ref CollectElements()
		 * Non blocking, all drivers can be launched simultaneously, but before continue to next simulation step, \b f->Finish() must be called
		 * \param signal immediate value of signal
		 **/
		void Drive(data_t signal);
	};

	/**
	 * \brief Scanner - scans acoustic pressure in each element and store it to out_file
	 * \note For best performance use only rotation around X coordinate (if possible).
	 * 
	 */
	struct scanner {
		field* f = nullptr; ///< pointer to acoustic field, where transducer exists
		size_t num_elements = 0; ///< number of elements of transducer
		cl::Buffer buff_elements; ///< coordinate of each element of transducer format: uint32 { x0, x1 ... xn, y1 .. yn, z1 .. zn }
		cl::Buffer buff_data; ///< pressure in each element of transducer (device-side), ordes is same as element-coordinations order
		std::ofstream out_file;
		uint32_t store_every_nth_frame;

		scanner() {};
		scanner(field &f, vec3<uint32_t> position, vec3<double> rotation, vec2<uint32_t> size, std::string out_file_name, uint32_t store_every_nth_frame = 1)
			{ Prepare(f, position, rotation, size, out_file_name, store_every_nth_frame); }
		// ~scanner() { if(out_file.is_open()) out_file.close(); } file is closed by it's destructor

		void Prepare(field &_f, vec3<uint32_t> position, vec3<double> rotation, vec2<uint32_t> size, std::string out_file_name, uint32_t _store_every_nth_frame = 1);

		/** \brief Scans pressure of elements into buffer on device
		 *
		 * Must be already initialized by \ref Prepare()
		 * Non blocking, all scanners can be launched simultaneously,
		 * but before continue to next step ( \ref Scan2file() ) \b f->Finish() or \b enqueueBarrierWithWaitList()
		 * must be called
		 **/
		void Scan2devmem();

		/**
		 * \brief copy data from device memory to output file - call it after \ref Scan2devmem()
		 */
		void Scan2file();
	};

	/** \brief Scanner - scans acoustic pressure in each element and store it to host's memory array */
	// struct scanner : transducer {
	// 	data_t* data = nullptr; ///< pressure in each element of transducer (host-side), ordes is same as elements order - see \ref transducer::GetElementsCoords()
	// 	cl::Buffer buff_data; ///< pressure in each element of transducer (device-side)

	// 	scanner() {};
	// 	scanner(field& f) : transducer(f) {}
	// 	~scanner() { if (data != nullptr) delete[] data; }

	// 	void CollectElements(); ///< Store coordinates of scanner's elements and allocates memory on host (data) and device (buff_data), see \ref transducer::CollectElements()
		
	// 	/** \brief Scans pressure of elements into buffer on device
	// 	 *
	// 	 * Coordinates of elements must be already initialized by \ref CollectElements()
	// 	 * Non blocking, all scanners can be launched simultaneously, but before continue to next step (\ref Scan2hostmem()), \b f->Finish() must be called
	// 	 **/
	// 	void Scan2devmem();
	// 	void Scan2hostmem(); ///< copy data from device to host - call it after \ref Scan2devmem()
	// };

};

#endif

