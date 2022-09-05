#include <iostream>
#include <fstream>
#include <cstdint>
#include <algorithm>
#include "object_creator.hpp"
#include "f3d_math.hpp"

using namespace f3d;

/*
STL file format (binary):
UINT8[80]    – Header                 - 80 bytes
UINT32       – Number of triangles    -  4 bytes
foreach triangle                      - 50 bytes:
    REAL32[3] – Normal vector             - 12 bytes
    REAL32[3] – Vertex 1                  - 12 bytes
    REAL32[3] – Vertex 2                  - 12 bytes
    REAL32[3] – Vertex 3                  - 12 bytes
    UINT16    – Attribute byte count      -  2 bytes
end
 */
object_3d_vi* loader::LoadSTL(const char* path, glm::vec3* model_max_dim)
{
    char rdata[80]; // biggest chunk of data is header (80 Bytes)
    uint32_t nr_of_triangles;
    glm::vec3 max_dim = {0.0f, 0.0f, 0.0f};
    std::ifstream data_file;
    object_3d_vi* model = new object_3d_vi;

    try
    {
        data_file.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        data_file.open(path, std::ios::binary);
        data_file.read(rdata, 80); // read header and flush it
        data_file.read((char*)&nr_of_triangles, 4); // assume little-endian machine (x86 is)
        model->vertices.reserve(nr_of_triangles * 3 * 3); // 3 vertices per triangle * 3 coordinates per vertex
        model->normals.reserve(nr_of_triangles * 3 * 3);
        // for each triangle
        for(uint32_t i = 0; i < nr_of_triangles; i++)
        {
            // load 3 vertices
            float vertex[9]; // x1, y1, z1, x2, y2 .. z3
            data_file.read(rdata, sizeof(float) * 3); // flush normal vector, we will calc it bellow
            data_file.read((char*)vertex, sizeof(float) * 9); // load all 3 vertices
            data_file.read(rdata, 2); // flush "Attribute byte count"
            model->vertices.insert(model->vertices.end(), vertex, vertex + 9);
            // calc normal vector
            glm::vec3 a = {vertex[0], vertex[1], vertex[2]}; // normal vector calculation instead use of nv from SLT file
            glm::vec3 b = {vertex[3], vertex[4], vertex[5]};
            glm::vec3 c = {vertex[6], vertex[7], vertex[8]};
            auto nv = glm::normalize(glm::cross(b - a, c - a));
            float norm_vec[] = {nv.x, nv.y, nv.z};
            model->normals.insert(model->normals.end(), norm_vec, norm_vec + 3); // same normal for each vertex in STL
            model->normals.insert(model->normals.end(), norm_vec, norm_vec + 3);
            model->normals.insert(model->normals.end(), norm_vec, norm_vec + 3);
            // check maximum positive dimmensions
            max_dim.x = std::max(max_dim.x, std::max(a.x, std::max(b.x, c.x)));
            max_dim.y = std::max(max_dim.y, std::max(a.y, std::max(b.y, c.y)));
            max_dim.z = std::max(max_dim.z, std::max(a.z, std::max(b.z, c.z)));
        }
        data_file.close();
    }
    catch(const std::exception& e)
    {
        std::string s = "ERROR::f3d::loader::LoadSTL(): FILE_NOT_SUCCESFULLY_READ: " + std::string(path) + ":\n";
        throw std::runtime_error(s);
    }

    if(model_max_dim != nullptr)
    {
        // update maximal dimensions
        model_max_dim->x = std::max(model_max_dim->x, max_dim.x);
        model_max_dim->y = std::max(model_max_dim->y, max_dim.y);
        model_max_dim->z = std::max(model_max_dim->z, max_dim.z);
    }

    return model;    
}

void object3d::Prepare( object_3d_vi* model,
                        glm::vec3 translation,
                        glm::vec3 rotation,
                        glm::vec3 scale,
                        glm::vec4 color )
{
    this->_translation = translation;
    this->_rotation = rotation;
    this->_scale = scale;
    this->_color = color;

    glGenVertexArrays(1, &vao);
	glBindVertexArray(vao); // bind Vertex Array Object
	glGenBuffers(2, buff);
	// vertices
	glBindBuffer(GL_ARRAY_BUFFER, buff[0]); // copy vertices array to buffer for OpenGL use
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * model->vertices.size(), model->vertices.data(), GL_STATIC_DRAW);
    // then set our vertex attributes pointers (tell OpenGL how to interpred vertex data)
    glVertexAttribPointer(
        0, // attribute "location = 0" in shader
        3, // size of attribute (in floats, vertex coord is vec3)
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float), // space between consecutive vertex attributes
        (void*)0 // offset of this attribute in vertices "structure"
    );
    glEnableVertexAttribArray(0);
    // normals
	glBindBuffer(GL_ARRAY_BUFFER, buff[1]); // copy vertices array to buffer for OpenGL use
	glBufferData(GL_ARRAY_BUFFER, sizeof(float) * model->normals.size(), model->normals.data(), GL_STATIC_DRAW);
    glVertexAttribPointer(
        1, // attribute "location = 1" in shader
        3, // size of attribute (in floats, normal vector is vec3)
        GL_FLOAT,
        GL_FALSE,
        3 * sizeof(float), // space between consecutive normal attributes
        (void*)0 // offset of this attribute in vertices "structure"
    );
    glEnableVertexAttribArray(1);
    nr_of_vertices = model->vertices.size();
    delete model; // no more needed, everythinh is in GPU
}

void object3d::Draw(const glm::mat4& view_matrix, const glm::vec3& light_pos)
{
    if(_shader == nullptr)
        return;
    
    _shader->use();
    // Create transformationmatrix from scale, rotation, and translation vectors
    auto trans_mat = glm::mat4(1.0f);
    trans_mat  = glm::translate(trans_mat, _translation);
    trans_mat *= f3d::RotationMatrix(_rotation);
    trans_mat  = glm::scale(trans_mat, _scale);
    auto normal_mat = glm::mat3(glm::transpose(glm::inverse(trans_mat))); // transformation matrix for normals
    _shader->setUniform("view", view_matrix);
    _shader->setUniform("transform", trans_mat);
    _shader->setUniform("normal_mat", normal_mat);
    _shader->setUniform("color", _color);
    _shader->setUniform("light_pos", light_pos);

    glBindVertexArray(vao);
    // glPolygonMode(GL_FRONT_AND_BACK,GL_LINE); // wireframe
    // glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
    glDrawArrays(GL_TRIANGLES, 0, nr_of_vertices);
}
