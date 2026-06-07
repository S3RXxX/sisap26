#include <iostream>
#include <vector>
#include <string>
#include <hdf5/serial/H5Cpp.h>   // Cambia a <H5Cpp.h> si da error

int main() {
    const std::string filePath = 
        "h5 file path"; //Ejecutar primero download_data y mirar el path donde se han descargado los datos!

    try {
        H5::H5File file(filePath, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet("train");

        H5::DataSpace dataspace = dataset.getSpace();

        int rank = dataspace.getSimpleExtentNdims();
        std::vector<hsize_t> dims(rank);
        dataspace.getSimpleExtentDims(dims.data());

        hsize_t num_vectors = dims[0];
        hsize_t dim = dims[1];
        hsize_t total_elements = num_vectors * dim;

        std::cout << "Cargando " << num_vectors << " vectores de dimensión " << dim << " (float16)..." << std::endl;

        // Leer datos como uint16_t (float16 en HDF5)
        std::vector<float> data_float16(total_elements);
        dataset.read(data_float16.data(), H5::PredType::NATIVE_FLOAT);

        // Convertir float16 → float32
        std::vector<float> embeddings(total_elements);

        for (hsize_t i = 0; i < total_elements; ++i) {
            // uint16_t h = data_float16[i];
            
            // uint32_t sign = (h & 0x8000u) << 16;
            // uint32_t exponent = (h & 0x7C00u) << 13;
            // uint32_t mantissa = (h & 0x03FFu) << 13;

            // uint32_t float_bits = sign | exponent | mantissa;

            // if (exponent == 0) {
            //     float_bits = sign | mantissa;  // subnormal o cero
            // }

            embeddings[i] = data_float16[i];
        }

        std::cout << "✅ ¡Datos cargados correctamente en std::vector<float>!" << std::endl;
        std::cout << "Total elementos: " << embeddings.size() << std::endl;

        // Mostrar muestra
        std::cout << "\nPrimer vector:" << std::endl;
        for (int j = 0; j < 1024; ++j) {
            std::cout << embeddings[j] << " ";
        }
        std::cout << std::endl;

        // Ejemplo de acceso:
        // float valor = embeddings[vector_idx * 1024 + componente];

    } catch (H5::Exception& err) {
        std::cerr << "❌ Error HDF5: " << err.getDetailMsg() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    return 0;
}