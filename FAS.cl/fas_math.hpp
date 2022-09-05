#ifndef FAS_MATH_H
#define FAS_MATH_H

#include <cmath>
#include "fas_data_types.hpp"

/** 
 * @file
 * @brief Fragments of linear algebra
 */

namespace fas {

	/**
	 * \brief Generates rotation matrix
	 *
	 * \param rot   Tait-Bryan rotation ( rot{ alpha, beta, gamma }, where: \b alpha around \b X3, \b beta around \b Y2, \b gamma around \b Z1 ) in [rad].
     *              Please note reversed order of individual rotations: first, object is rotated around \b Z than \b Y' and at the end around \b X''.
	 * \return      transformation matrix for Mat x Vec multiply
	 */
	template<typename T>
	inline mat3_3<T> RotationMatrix(const vec3<double>& rot) {
		mat3_3<T> res;

		double c1 = cos(rot.z); // rot vector uses double - better precision in common angles (eg. pi/2)
		double c2 = cos(rot.y);
		double c3 = cos(rot.x);
		double s1 = sin(rot.z);
		double s2 = sin(rot.y);
		double s3 = sin(rot.x);
		res.a11 = (T)(c1 * c2);
		res.a12 = (T)(c1 * s2 * s3 - c3 * s1);
		res.a13 = (T)(s1 * s3 + c1 * s2 * c3);
		res.a21 = (T)(s1 * c2);
		res.a22 = (T)(c1 * c3 + s1 * s2 * s3);
		res.a23 = (T)(s1 * s2 * c3 - c1 * s3);
		res.a31 = (T)(-s2);
		res.a32 = (T)(c2 * s3);
		res.a33 = (T)(c2 * c3);

		return res;
	}

}

#endif
