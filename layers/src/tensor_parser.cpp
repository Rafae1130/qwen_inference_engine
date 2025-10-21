
#include <iostream>

#include "tensor_parser.hh"
#include "json.hh"

using json = nlohmann::json;

#include <unordered_map>


void from_json(const json& j, tensor& p) {
    j.at("shape").get_to(p.shape);
    j.at("data_offsets").get_to(p.data_offsets);

}


std::ostream& operator<<(std::ostream& os, const tensor& t) {
    os << "Tensor: " << t.tensor_name << "\n";
    os << "  layer: " << t.layer_index << "\n";
    os << "  short_name: " << t.short_name << "\n";
    os << "  shape: [ ";
    for (auto s : t.shape) os << s << " ";
    os << "]\n";
    os << "  offsets: [ " << t.data_offsets[0] << ", " << t.data_offsets[1] << " ]\n";
    return os;
}


std::vector<tensor> parsed_tensors(){

    std::vector<tensor> all_tensors;
    std::ofstream meta_out("../model_files/meta_data.txt");
    size_t global_offset = 0;

    std::vector<std::string> files = {
        "/mnt/data/rafaedata/Qwen3-14B/model-00001-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00002-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00003-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00004-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00005-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00006-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00007-of-00008.safetensors",
        "/mnt/data/rafaedata/Qwen3-14B/model-00008-of-00008.safetensors"
      };
    
    // std::vector<char> buffer(10*1024*1024);
    // std::ofstream out("/mnt/data/rafae/qwen_weights/weights.bin", std::ios::binary | std::ios::trunc);
    
    for (const auto& safe_tensor_name : files){
        std::ifstream f(safe_tensor_name,  std::ios::binary);
        uint64_t header_len;
        // in each safetensor file the first 8 bytes are the size of header
        f.read(reinterpret_cast<char*>(&header_len), sizeof(header_len));
        if (!f) {
            std::cout << "Failed to read JSON header (" << header_len << " bytes)\n";
 
        }


        // creating a string equal to the read header_len
        std::string json_data(header_len, '\0');
        
        // reading data equal to header len. i.e. complete metadata header
        f.read(json_data.data(), header_len);
        json j = json::parse(json_data);


        // the key is the layer name
        for (auto it = j.begin(); it != j.end(); ++it) {
            const std::string& key = it.key();

            if (key.rfind("model.", 0) == 0) { // key starts with "model."
                tensor t;
                //it.value gives the value associated with the key, 
                //and .get<tensor> convert the json value to our tensor struct.
                t = it.value().get<tensor>();
                t.tensor_name = key;

                // extracting layer index and short name 
                // find the starting index of layer in the key. which is the model name
                size_t layer_pos = key.find("layers.");
                if (layer_pos != std::string::npos) {
                    //find the dot after the layer number
                    size_t dot_after_layer = key.find('.', layer_pos + 7);
                    t.layer_index = std::stoi(key.substr(layer_pos + 7, dot_after_layer - (layer_pos + 7)));
                    t.short_name = key.substr(dot_after_layer + 1); // remove "model.layers.N."
                } else {
                    t.short_name = key.substr(6); // remove "model."
                }


                size_t tensor_size = t.data_offsets[1] - t.data_offsets[0];
                t.data_offsets[0] = global_offset;
                t.data_offsets[1] = global_offset + tensor_size;

                all_tensors.push_back(std::move(t));
                global_offset += tensor_size;
            }

            else if (key.rfind("lm_", 0) == 0) { // key starts with "model."
                tensor t;

                t = it.value().get<tensor>();
                t.tensor_name = key;
                t.short_name = "logits";

                size_t tensor_size = t.data_offsets[1] - t.data_offsets[0];
                t.data_offsets[0] = global_offset;
                t.data_offsets[1] = global_offset + tensor_size;

                all_tensors.push_back(std::move(t));
                global_offset += tensor_size;
            }
            
        }
        // while (f) {
        //     f.read(buffer.data(), buffer.size());
        //     out.write(buffer.data(), f.gcount());
        // }
    }

    for (const auto& t : all_tensors) {
        meta_out << t << "\n";   // uses operator<<
    }

    return all_tensors;
}


std::unordered_map<std::string, std::vector<tensor>> build_indexed_tensors() {
    auto all = parsed_tensors();
    std::unordered_map<std::string, std::vector<tensor>> indexed;

    for (auto& t : all) {
        if (t.layer_index >= 0) {
            // for each key there will be multiple tensor values depending on the index. if the size of map for the particular key is less than the new tensors index, then increase the size by 1.
            if (indexed[t.short_name].size() <= static_cast<size_t>(t.layer_index)) {
                indexed[t.short_name].resize(t.layer_index + 1);
            }
            indexed[t.short_name][t.layer_index] = t;
        } else {
            // special cases
            if (t.short_name == "embed_tokens.weight" ||
                t.short_name == "norm.weight" ||
                t.short_name == "logits"
            ) 
            {
                if (indexed[t.short_name].empty()) {
                    indexed[t.short_name].resize(1);
                }
                indexed[t.short_name][0] = t;
            }            
            else {
                // fallback for other non-layer tensors
                if (indexed[t.short_name].empty()) {
                    indexed[t.short_name].resize(1);
                }
                indexed[t.short_name][0] = t;
            }
        }
    }
    return indexed;
}




