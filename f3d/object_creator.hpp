#ifndef F3D_OBJECT_CREATOR_HPP
#define F3D_OBJECT_CREATOR_HPP

#include <vector>
#include "glm/glm.hpp"
#include "shader.hpp"

namespace f3d {

    // 3D object with vertices and indices
    struct object_3d_vi
    {
        std::vector<float> vertices; // format: {x, y, z} for each vertice
        std::vector<float> normals; // normal vector for each vertice, format: {x, y, z}
    };
    
    namespace loader {

        /**
         * @brief Loads 3D object from binary STL file
         * 
         * @param path              input file
         * @param model_max_dim*    updates maximal size of model in each dimmension (positive only)
         * @return object_3d_vi*    Pointer to newly created 3D model, caller must deallocate memory!
         */
        object_3d_vi* LoadSTL(const char* path, glm::vec3* model_max_dim = nullptr);

    }

    struct object3d
    {

        // data
		glm::vec3 _translation;
		glm::vec3 _rotation;
		glm::vec3 _scale;
        glm::vec4 _color;

		// OpenGL
		shader* _shader = nullptr;
		unsigned int vao; // Vertex Array Object ID
		unsigned int buff[2] = {0, 0}; // vertices, normals
	    size_t nr_of_vertices = 0; // total nr of vertices in final model

        object3d() {}
		object3d(shader& object_shader) { _shader = &object_shader; }

		object3d(   shader& object_shader,
                    object_3d_vi* model,
                    glm::vec3 translation = {0.0f, 0.0f, 0.0f},
                    glm::vec3 rotation = {0.0f, 0.0f, 0.0f},
                    glm::vec3 scale = {1.0f, 1.0f, 1.0f},
                    glm::vec4 color = {0.8f, 0.8f, 0.8f, 1.0f})
		{
            _shader = &object_shader;
			Prepare(model, translation, rotation, scale, color);
		}

        ~object3d() { /*glDeleteBuffers(2, buff);*/ }

        /**
         * @brief Loads model's data to GPU
         * 
         * @param model         pointer to 3D model, will be deallocated here after copy to GPU
         * @param translation 
         * @param rotation 
         * @param scale 
         * @param color 
         */
		void Prepare(object_3d_vi* model,
                    glm::vec3 translation = {0.0f, 0.0f, 0.0f},
                    glm::vec3 rotation = {0.0f, 0.0f, 0.0f},
                    glm::vec3 scale = {1.0f, 1.0f, 1.0f},
                    glm::vec4 color = {0.8f, 0.8f, 0.8f, 1.0f});

        void Draw(const glm::mat4& view_matrix, const glm::vec3& light_pos = {0,0,0});

        //glm::vec3 GetSize();
    };
    
}

#endif
