#ifndef NULL
	#define NULL 0
#endif
#define INDEX2D(x, y) ((size_t)(y) * (size_t)x_size + (size_t)(x))
#define INDEX3D(x, y, z) ((size_t)(z) * (size_t)x_size * (size_t)y_size + (size_t)(y) * (size_t)x_size + (size_t)(x))

/*******************************/
/* Acoustic field calculations */
/*******************************/

// run this kernel in 3D range { field.size.x, field.size.y, field.size.z }
kernel void clear ( global float * pressure, global float * pressure_tm1, global uchar * material, global float * rms_sum ) {

	size_t x_size = get_global_size(0);
	size_t y_size = get_global_size(1);
	size_t my_idx = INDEX3D(get_global_id(0), get_global_id(1), get_global_id(2));

	pressure [my_idx] = 0.0f;
	pressure_tm1 [my_idx] = 0.0f;
	material[my_idx] = 0;
	if (rms_sum != NULL)
		rms_sum[my_idx] = 0.0f;
}

// run this kernel in 3D range { field.size.x, field.size.y, field.size.z }
kernel void sim_step (	global const float * p_t, // field.A/B (see C++ source)
						global float * p_tm1, // field.A/B (see C++ soucre)
						global const uchar * material, // field.buff_mat
						constant float * r, // arrays[256] of material properties
						constant float * c, // ...
						// TODO: use k = dt*dt/(dx*dx) instead of following two params
						float dx, // edge of cubic elements [m]
						float dt // time step [s]
						) {

	size_t x_size = get_global_size(0);
	size_t y_size = get_global_size(1);
	size_t z_size = get_global_size(2);

	size_t my_z = get_global_id(2);
	size_t my_y = get_global_id(1);
	size_t my_x = get_global_id(0);

	// don't calc boundaries 
	// TODO: pevne okrajove podminky -> nesmysl, potřeba použít volné -> ideální odraz od pevné překážky ( pravda ?? -> nutno konzultovat )
	if (my_z > 0 && my_z < z_size - 1 && \
		my_y > 0 && my_y < y_size - 1 && \
		my_x > 0 && my_x < x_size - 1)
	{

		size_t my_idx = INDEX3D(my_x, my_y, my_z);
		float my_r = r[material[my_idx]]; // characteristic acoustic impedance of my element
		float my_c = c[material[my_idx]]; // my speed of sound
		float my_p = p_t[my_idx]; // actual pressure of my element

		// characteristic acoustic impedance of adjacent elements
		float xp1_r = r[material[my_idx + 1]];
		float xm1_r = r[material[my_idx - 1]];
		float yp1_r = r[material[my_idx + x_size]];
		float ym1_r = r[material[my_idx - x_size]];
		float zp1_r = r[material[my_idx + x_size * y_size]];
		float zm1_r = r[material[my_idx - x_size * y_size]];

		// actual pressure in adjacent elements
		float xp1_p = p_t[my_idx + 1];
		float xm1_p = p_t[my_idx - 1];
		float yp1_p = p_t[my_idx + x_size];
		float ym1_p = p_t[my_idx - x_size];
		float zp1_p = p_t[my_idx + x_size * y_size];
		float zm1_p = p_t[my_idx - x_size * y_size];

		float acc;
		// calc influence of neighboring elements:
		// transmission wave from neighboring elements ( Tcoef = 2*r_my / (r_my + r_neigh) )
		acc =  2.0f * my_r / (my_r + xp1_r) * xp1_p; // "* 2.0f" can be moved to wave eq bellow
		acc += 2.0f * my_r / (my_r + yp1_r) * yp1_p;
		acc += 2.0f * my_r / (my_r + zp1_r) * zp1_p;
		acc += 2.0f * my_r / (my_r + xm1_r) * xm1_p;
		acc += 2.0f * my_r / (my_r + ym1_r) * ym1_p;
		acc += 2.0f * my_r / (my_r + zm1_r) * zm1_p;
		
		// myself influence - reflected wave from boundary (0 if same material: r_my = r_neigh)
		acc += (xp1_r - my_r) / (xp1_r + my_r) * my_p;
		acc += (yp1_r - my_r) / (yp1_r + my_r) * my_p;
		acc += (zp1_r - my_r) / (zp1_r + my_r) * my_p;
		acc += (xm1_r - my_r) / (xm1_r + my_r) * my_p;
		acc += (ym1_r - my_r) / (ym1_r + my_r) * my_p;
		acc += (zm1_r - my_r) / (zm1_r + my_r) * my_p;

		// from now, for this element, p_tm1 will be p(t + 1); after finish of this simulation-step, buffers will be swapped by host code
		p_tm1[my_idx] = dt * dt * my_c * my_c / (dx * dx) * (acc - 6.0f * my_p) + 2.0f * my_p - p_tm1[my_idx];
	}
}

// run this kernel in 3D range { field.size.x, field.size.y, field.size.z }
// "integrates" actual-value of pressure
kernel void rms_sum (	global const float* p_t,
						global float* sum,
						float window
						) {
	size_t x_size = get_global_size(0);
	size_t y_size = get_global_size(1);
	size_t my_idx = INDEX3D(get_global_id(0), get_global_id(1), get_global_id(2));

	float my_p_ = p_t[my_idx] * window; // "windowed" pressure
	sum[my_idx] += my_p_ * my_p_; // p^2
}

// run this kernel in 3D range { field.size.x, field.size.y, field.size.z }
kernel void rms_final(	global float* sum,
						float inv_steps, // 1/(total number of simulation steps)
						float inv_sqrt_nnpg // 1/sqrt(nnpg)
						) {
	size_t x_size = get_global_size(0);
	size_t y_size = get_global_size(1);
	size_t my_idx = INDEX3D(get_global_id(0), get_global_id(1), get_global_id(2));

	sum[my_idx] = inv_sqrt_nnpg * sqrt(inv_steps * sum[my_idx]);
}

/********************************/
/* Acoustic field 2D/3D objects */
/********************************/

// run this kernel in 2D range { 2 * rect.size.x, 2 * rect.size.y }
// draw filled rectangle into 3D array - sets coresponding elements of mat_arr to material
kernel void object_rect (	global uchar * mat_arr, // field.buff_mat
							constant float * rot_pos, // position and rotation: elements [0..8] rotation matrix, [9..11] position
							uint x_size, uint y_size, uint z_size, // size of acoustic FIELD (mat_arr array)
							uchar material // material of object
							) {

	// scale by 0.5 (to remove "holes" in rotated object caused by float to int conversion)
	float my_xf = 0.5f * get_global_id(0);
	float my_yf = 0.5f * get_global_id(1);
	float my_zf = 0.0f;

	// rotate & translate
	float my_xff = rot_pos[0] * my_xf + rot_pos[1] * my_yf + rot_pos[9]; // x
	float my_yff = rot_pos[3] * my_xf + rot_pos[4] * my_yf + rot_pos[10]; // y
	float my_zff = rot_pos[6] * my_xf + rot_pos[7] * my_yf + rot_pos[11]; // z

	// back to integer
	uint my_x = my_xff;
	uint my_y = my_yff;
	uint my_z = my_zff;

	// field's boundary check
	if( my_xff >= 0.0f && my_x < x_size &&
		my_yff >= 0.0f && my_y < y_size &&
		my_zff >= 0.0f && my_z < z_size ) {
		// store
		size_t my_idx = INDEX3D(my_x, my_y, my_z);
		mat_arr[my_idx] = material;
	}
}

// run this kernel in 2D range { 4 * ellipse.radius.a, 4 * ellipse.radius.b }
// draw filled ellipse into 3D array - sets coresponding elements of mat_arr to material
kernel void object_ellipse (	global uchar * mat_arr, // field.buff_mat
								constant float * rot_pos, // position and rotation: elements [0..8] rotation matrix, [9..11] position
								uint x_size, uint y_size, uint z_size, // size of acoustic FIELD (mat_arr array)
								uchar material // material of object
								) {

	float radius_a = 0.25f * get_global_size(0);
	float radius_b = 0.25f * get_global_size(1);

	// scale by 0.5 (to remove "holes" in rotated object caused by float to int conversion)
	float my_xf = 0.5f * get_global_id(0) - radius_a;
	float my_yf = 0.5f * get_global_id(1) - radius_b;
	float my_zf = 0.0f;

	// test if my coords are within ellipse (ellipse equation)
	if( my_xf*my_xf/(radius_a*radius_a) + my_yf*my_yf/(radius_b*radius_b) <= 1.0f ) {

		// rotate & translate
		float my_xff = rot_pos[0] * my_xf + rot_pos[1] * my_yf + rot_pos[9]; // x
		float my_yff = rot_pos[3] * my_xf + rot_pos[4] * my_yf + rot_pos[10]; // y
		float my_zff = rot_pos[6] * my_xf + rot_pos[7] * my_yf + rot_pos[11]; // z

		// back to integer
		uint my_x = my_xff;
		uint my_y = my_yff;
		uint my_z = my_zff;

		// field's boundary check
		if( my_xff >= 0.0f && my_x < x_size &&
			my_yff >= 0.0f && my_y < y_size &&
			my_zff >= 0.0f && my_z < z_size ) {
			// store
			size_t my_idx = INDEX3D(my_x, my_y, my_z);
			mat_arr[my_idx] = material;
		}
	}
}

// run this kernel in 3D range { 2 * box.size.x, 2 * box.size.y, 2 * box.size.z }
// draw filled box into 3D array - sets coresponding elements of mat_arr to material
kernel void object_box (	global uchar * mat_arr, // field.buff_mat
							constant float * rot_pos, // position and rotation: elements [0..8] rotation matrix, [9..11] position
							uint x_size, uint y_size, uint z_size, // size of acoustic FIELD (mat_arr array)
							uchar material // material of object
							) {

	// scale by 0.5 (to remove "holes" in rotated object caused by float to int conversion)
	float my_xf = 0.5f * get_global_id(0);
	float my_yf = 0.5f * get_global_id(1);
	float my_zf = 0.5f * get_global_id(2);

	// rotate & translate
	float my_xff = rot_pos[0] * my_xf + rot_pos[1] * my_yf + rot_pos[2] * my_zf + rot_pos[9]; // x
	float my_yff = rot_pos[3] * my_xf + rot_pos[4] * my_yf + rot_pos[5] * my_zf + rot_pos[10]; // y
	float my_zff = rot_pos[6] * my_xf + rot_pos[7] * my_yf + rot_pos[8] * my_zf + rot_pos[11]; // z

	// back to integer
	uint my_x = my_xff;
	uint my_y = my_yff;
	uint my_z = my_zff;

	// field's boundary check
	if( my_xff >= 0.0f && my_x < x_size &&
		my_yff >= 0.0f && my_y < y_size &&
		my_zff >= 0.0f && my_z < z_size ) {
		// store
		size_t my_idx = INDEX3D(my_x, my_y, my_z);
		mat_arr[my_idx] = material;
	}
}

// run this kernel in 2D range { 4 * base.radius.a, 4 * base.radius.b }
// draw filled elliptical cylinder into 3D array - sets coresponding elements of mat_arr to material
kernel void object_cylinder (	global uchar * mat_arr, // field.buff_mat
							constant float * rot_pos, // position and rotation: elements [0..8] rotation matrix, [9..11] position
							uint x_size, uint y_size, uint z_size, // size of acoustic FIELD (mat_arr array)
							uchar material, // material of object
							uint two_height // 2 * height of cylinder
							) {

	float radius_a = 0.25f * get_global_size(0);
	float radius_b = 0.25f * get_global_size(1);

	// scale by 0.5 (to remove "holes" in rotated object caused by float to int conversion)
	float my_xf = 0.5f * get_global_id(0) - radius_a;
	float my_yf = 0.5f * get_global_id(1) - radius_b;
	float my_zf = 0.0f;

	// test if my coords are within object's base (x,y) (ellipse equation)
	if( my_xf*my_xf/(radius_a*radius_a) + my_yf*my_yf/(radius_b*radius_b) <= 1.0f ) {

		for( uint i = 0; i < two_height; i++, my_zf += 0.5f ) {
			// rotate & translate
			float my_xff = rot_pos[0] * my_xf + rot_pos[1] * my_yf + rot_pos[2] * my_zf + rot_pos[9]; // x
			float my_yff = rot_pos[3] * my_xf + rot_pos[4] * my_yf + rot_pos[5] * my_zf + rot_pos[10]; // y
			float my_zff = rot_pos[6] * my_xf + rot_pos[7] * my_yf + rot_pos[8] * my_zf + rot_pos[11]; // z

			// back to integer
			uint my_x = my_xff;
			uint my_y = my_yff;
			uint my_z = my_zff;

			// field's boundary check
			if( my_xff >= 0.0f && my_x < x_size &&
				my_yff >= 0.0f && my_y < y_size &&
				my_zff >= 0.0f && my_z < z_size ) {
				// store
				size_t my_idx = INDEX3D(my_x, my_y, my_z);
				mat_arr[my_idx] = material;
			}
		}
	}
}

// run this kernel in 3D range { 4 * ellipsoid.radius.a, 4 * ellipsoid.radius.b, 4 * ellipsoid.radius.c }
// draw filled ellipsoid into 3D array - sets coresponding elements of mat_arr to material
kernel void object_ellipsoid (	global uchar * mat_arr, // field.buff_mat
							constant float * rot_pos, // position and rotation: elements [0..8] rotation matrix, [9..11] position
							uint x_size, uint y_size, uint z_size, // size of acoustic FIELD (mat_arr array)
							uchar material // material of object
							) {

	float radius_a = 0.25f * get_global_size(0);
	float radius_b = 0.25f * get_global_size(1);
	float radius_c = 0.25f * get_global_size(2);

	// scale by 0.5 (to remove "holes" in rotated object caused by float to int conversion)
	float my_xf = 0.5f * get_global_id(0) - radius_a;
	float my_yf = 0.5f * get_global_id(1) - radius_b;
	float my_zf = 0.5f * get_global_id(2) - radius_c;

	// test if my coords are within object (ellipsoid equation)
	if( my_xf*my_xf/(radius_a*radius_a) + my_yf*my_yf/(radius_b*radius_b) + my_zf*my_zf/(radius_c*radius_c) <= 1.0f ) {

		// rotate & translate
		float my_xff = rot_pos[0] * my_xf + rot_pos[1] * my_yf + rot_pos[2] * my_zf + rot_pos[9]; // x
		float my_yff = rot_pos[3] * my_xf + rot_pos[4] * my_yf + rot_pos[5] * my_zf + rot_pos[10]; // y
		float my_zff = rot_pos[6] * my_xf + rot_pos[7] * my_yf + rot_pos[8] * my_zf + rot_pos[11]; // z

		// back to integer
		uint my_x = my_xff;
		uint my_y = my_yff;
		uint my_z = my_zff;

		// field's boundary check
		if( my_xff >= 0.0f && my_x < x_size &&
			my_yff >= 0.0f && my_y < y_size &&
			my_zff >= 0.0f && my_z < z_size ) {
			// store
			size_t my_idx = INDEX3D(my_x, my_y, my_z);
			mat_arr[my_idx] = material;
		}
	}
}

/*********************************/
/* Transducer - driver / scanner */
/*********************************/

// run this kernel in 2D range { field.size.y, field.size.z }
// counts number of transducer's elements (with MSB set) in each line along the x-axis
kernel void tdcr_count_elements (	global const uchar * mat_arr, // field.buff_mat array
									global uint * count_mat, // array of size { field.size.y, field.size.z }, drv elements count - output of kernel
									uint x_size // field.size.x [elements]
									) {
	size_t my_y = get_global_id(0);
	size_t my_z = get_global_id(1);
	size_t y_size = get_global_size(0);

	uint counter = 0;
	for( uint my_x = 0; my_x < x_size; my_x++ ) {
		if( mat_arr[INDEX3D( my_x, my_y, my_z )] & 0x80 )
			counter++; // MSB is set, inc counter
	}

	// store result
	count_mat[my_y + my_z * y_size] = counter;
}

// run this kernel in 2D range { field.size.y, field.size.z }
// store coordinates of transducer's elements (with MSB set) in each line along the x-axis
kernel void tdcr_collect_elements (	global const uchar * mat_arr, // field.buff_mat array with MSBs set in transducer's elements
										global const ulong * psum, // output of prefix sum kernels { field.size.y, filed.size.z }
										global uint * elements, // array of element's coordinates, format: { x0, x1 ... xn, y1 .. yn, z1 .. zn } - output of kernel
										uint x_size, // field.size.x [elements]
										ulong no_elements // total number of transducer's elements ( = psum[last])
										) {
	size_t my_y = get_global_id(0);
	size_t my_z = get_global_id(1);
	size_t y_size = get_global_size(0);
	size_t offset; // offset of first my element in elements array
	if( my_y == 0 && my_z == 0 )
		offset = 0;
	else
		offset = psum[my_y + my_z * y_size - 1];

	for( uint my_x = 0; my_x < x_size; my_x++ ) {
		if( mat_arr[INDEX3D( my_x, my_y, my_z )] & 0x80 ) {
			// store coordinates
			elements[offset] = my_x;
			elements[offset + no_elements] = (uint)my_y;
			elements[offset + 2 * no_elements] = (uint)my_z;
			offset++;
		}
	}
}

// run this kernel in 3D range { field.size.x, field.size.y, field.size.z }
// clears MSB bits in all field.buff_mat (MSB is used for transducer creation)
kernel void tdcr_clear_mat_MSBs ( global uchar * mat ) {
	size_t x_size = get_global_size(0);
	size_t y_size = get_global_size(1);

	mat[INDEX3D(get_global_id(0), get_global_id(1), get_global_id(2))] &= 0x7F;
}

// run this kernel in 1D range { elements.number_of_elements }
// sets pressure in each element of transducer to value of signal
kernel void drive ( global float* p_t,
					float signal,
					global const uint * elements, // coordinates of transducer's elements, format: { x0, x1 ... xn, y1 .. yn, z1 .. zn }
					uint x_size, uint y_size // field.size
					) {

	size_t offset = get_global_size(0); // begin of y part of coordinates; 2*offset => begin of z part of coordinates
	size_t my_idx = get_global_id(0);
	size_t my_x = elements[my_idx];
	size_t my_y = elements[my_idx + offset];
	size_t my_z = elements[my_idx + 2 * offset];

	p_t[INDEX3D(my_x, my_y, my_z)] = signal;
}

// run this kernel in 1D range { elements.number_of_elements }
// scans pressure in each element and store it to p_out array
kernel void scan ( 	global const float * p_t,
					global const uint * elements, // coordinates of transducer's elements, format: { x0, x1 ... xn, y1 .. yn, z1 .. zn }
					global float * p_out, // pressure in each element, output of kernel
					uint x_size, uint y_size // field.size
					) {
	
	size_t offset = get_global_size(0); // begin of y part of coordinates; 2*offset -> begin of z part of coordinates
	size_t my_idx = get_global_id(0);
	size_t my_x = elements[my_idx];
	size_t my_y = elements[my_idx + offset];
	size_t my_z = elements[my_idx + 2 * offset];

	p_out[my_idx] = p_t[INDEX3D(my_x, my_y, my_z)];
}

/********************/
/* Helper functions */
/********************/

// 1. PART OF ALGORITHM
// run this kernel in 1D range { in.size.y }
// calculates prefix-sum around EACH ROW SEPARATELY (around x axis) of 2D array and stores it to element { x-1, y }
// so in "arr" array on position { x, y } is stored prefix sum of { x, y } + value of { x, y }
// last_col 1D array where sums of whole rows are stored; it is copy of last column of "arr" array
kernel void horizontal_prefix_sum_uint_ulong (	global const uint * in, // input 2D array from which prefix sum is calculated
												global ulong * arr, // input to part 2
												global ulong * last_col, // input to part 2
												uint x_size // in.size.x
												) {
	uint my_y = get_global_id(0); // cast to int, bigger than 2^32 is not allowed
	ulong sum = 0;

	for( uint my_x = 0; my_x < x_size; my_x++ )	{
		size_t idx = INDEX2D( my_x, my_y );
		sum += in[idx];
		arr[idx] = sum;
	}

	last_col[my_y] = sum; // store sums of whole rows into separate 1D field - used by next kernel
}

// 2. PART OF ALGORITHM
// run this kernel in 1D range { arr.size.x }
// second part of prefix sum procedure
// to each column in row, add value of { previous row, last column }
// after run of both kernels: in each element of arr is sum of it's original value and original values of all preceding elements (dowto {0,0})
kernel void vertical_prexix_sum_ulong ( global ulong * arr, // out from 1. part; output from this kernel
										global const ulong * last_col, // from 1. part
										uint y_size // arr.size.y
										) {
	uint my_x = get_global_id(0); // cast to int, bigger than 2^32 is not allowed
	uint x_size = get_global_size(0);
	local ulong prev_row_last_col; // shared variable

	// first row leave as is
	for( uint my_y = 1; my_y < y_size; my_y++ ) {
		size_t my_idx = INDEX2D(my_x, my_y);

		if( get_local_id(0) == 0 )
			prev_row_last_col = last_col[my_y - 1]; // if I am first in work-group, load shared value
		barrier(CLK_LOCAL_MEM_FENCE);
		arr[my_idx] += prev_row_last_col; // add to my element value of last element in previous row (here is sum of whole prev. row)
	}
}
