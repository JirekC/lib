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
	 * \param rot Tait-Bryan rotation ( rot{ alpha, beta, gamma }, where: \b alpha around Z1, \b beta around Y2, \b gamma around X3 ) in [rad]
	 * \return transformation matrix for Mat x Vec multiply
	 */
	template<typename T>
	inline mat3_3<T> RotationMatrix(const vec3<double>& rot) {
		mat3_3<T> res;

		double c1 = cos(rot.x); // rot vector uses double - better precision in common angles (eg. pi/2)
		double c2 = cos(rot.y);
		double c3 = cos(rot.z);
		double s1 = sin(rot.x);
		double s2 = sin(rot.y);
		double s3 = sin(rot.z);
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
