#include <vulkan/vulkan.h>
#include <vector>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include "build/levelize.comp.spv.h" // Include the compiled shader bytecode

int main(int argc, char** argv) {

    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <input_image_path>" << std::endl;
        return -1;
    }
    const char* inputImagePath = argv[1];
    printf("Input image path: %s\n", inputImagePath);
    int imageWidth, imageHeight, imageChannels;
    unsigned char* imageData = stbi_load(inputImagePath, &imageWidth, &imageHeight, &imageChannels, STBI_rgb_alpha);
    if (!imageData) {
        std::cerr << "Failed to load image: " << inputImagePath << std::endl;
        return -1;
    }
    std::cout << "Image loaded: " << imageWidth << "x" << imageHeight << " (" << imageChannels << " channels)" << std::endl;

    // Initialize Vulkan and create a compute pipeline
    VkInstance instance;
    VkPhysicalDevice physicalDevice = VK_NULL_HANDLE;
    VkDevice device;
    VkQueue computeQueue;
    uint32_t queueFamilyIndex = UINT32_MAX;
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Halftoning";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_0;
    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    vkCreateInstance(&createInfo, nullptr, &instance);
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    physicalDevice = devices[0];
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());
    for (uint32_t i = 0; i < queueFamilyCount; i++) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            queueFamilyIndex = i;
            break;
        }
    }
    if (queueFamilyIndex == UINT32_MAX) {
        std::cerr << "Failed to find a compute queue family!" << std::endl;
        return -1;
    }
    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{};
    queueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;
    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;
    vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device);
    vkGetDeviceQueue(device, queueFamilyIndex, 0, &computeQueue);
    VkPhysicalDeviceProperties deviceProperties;
    vkGetPhysicalDeviceProperties(physicalDevice, &deviceProperties);
    std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;
    std::cout << "API Version: " << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_PATCH(deviceProperties.apiVersion) << std::endl;
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);
    VkDeviceSize bufferSize = imageWidth * imageHeight * 4; // RGBA

    VkBufferCreateInfo bufferCreateInfo{};
    bufferCreateInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferCreateInfo.size = bufferSize;
    bufferCreateInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferCreateInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    
    VkBuffer inputBuffer;
    vkCreateBuffer(device, &bufferCreateInfo, nullptr, &inputBuffer);
    VkBuffer outputBuffer;
    vkCreateBuffer(device, &bufferCreateInfo, nullptr, &outputBuffer);
    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, inputBuffer, &memRequirements);
    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    uint32_t memoryTypeIndex = UINT32_MAX;
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((memRequirements.memoryTypeBits & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            memoryTypeIndex = i;
            break;
        }
    }
    allocInfo.memoryTypeIndex = memoryTypeIndex;

    // Map memory and copy image data
    VkDeviceMemory inputBufferMemory;
    VkDeviceMemory outputBufferMemory;
    vkAllocateMemory(device, &allocInfo, nullptr, &inputBufferMemory);
    vkAllocateMemory(device, &allocInfo, nullptr, &outputBufferMemory);
    vkBindBufferMemory(device, inputBuffer, inputBufferMemory, 0);
    vkBindBufferMemory(device, outputBuffer, outputBufferMemory, 0);
    void* input_data;
    void* output_data;
    vkMapMemory(device, inputBufferMemory, 0, bufferSize, 0, &input_data);
    vkMapMemory(device, outputBufferMemory, 0, bufferSize, 0, &output_data);

    // Copy image_data (CPU) to Vulkan buffer (GPU) and free image data
    memcpy(input_data, imageData, bufferSize);
    stbi_image_free(imageData);

    // Create a command "pool"
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = queueFamilyIndex;
    VkCommandPool commandPool;
    vkCreateCommandPool(device, &poolInfo, nullptr, &commandPool);

    // Create a command "buffer"
    VkCommandBufferAllocateInfo cmdBufAllocInfo{};
    cmdBufAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdBufAllocInfo.commandPool = commandPool;
    cmdBufAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdBufAllocInfo.commandBufferCount = 1;
    VkCommandBuffer commandBuffer;
    vkAllocateCommandBuffers(device, &cmdBufAllocInfo, &commandBuffer);

    // Begin recording commands
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(commandBuffer, &beginInfo);

    // Create shader module
    VkShaderModuleCreateInfo shaderModuleCreateInfo{};
    shaderModuleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shaderModuleCreateInfo.codeSize = sizeof(build_levelize_comp_spv);
    shaderModuleCreateInfo.pCode = reinterpret_cast<const uint32_t*>(build_levelize_comp_spv);
    VkShaderModule shaderModule;
    vkCreateShaderModule(device, &shaderModuleCreateInfo, nullptr, &shaderModule);

    // Create descriptor set layout
    VkDescriptorSetLayoutBinding bindings[2] = {};
    bindings[0].binding = 0;
    bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    bindings[1].binding = 1;
    bindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo descriptorSetLayoutCreateInfo{};
    descriptorSetLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptorSetLayoutCreateInfo.bindingCount = 2;
    descriptorSetLayoutCreateInfo.pBindings = bindings;
    VkDescriptorSetLayout descriptorSetLayout;
    vkCreateDescriptorSetLayout(device, &descriptorSetLayoutCreateInfo, nullptr, &descriptorSetLayout);

    // Create pipeline layout with push constants
    VkPushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(uint32_t) * 3 + sizeof(float) * 32; // 3 uints + 8 mat2 (32 floats)
    
    VkPipelineLayoutCreateInfo pipelineLayoutCreateInfo{};
    pipelineLayoutCreateInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipelineLayoutCreateInfo.setLayoutCount = 1;
    pipelineLayoutCreateInfo.pSetLayouts = &descriptorSetLayout;
    pipelineLayoutCreateInfo.pushConstantRangeCount = 1;
    pipelineLayoutCreateInfo.pPushConstantRanges = &pushConstantRange;
    VkPipelineLayout pipelineLayout;
    vkCreatePipelineLayout(device, &pipelineLayoutCreateInfo, nullptr, &pipelineLayout);

    // Create compute pipeline
    VkComputePipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineCreateInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineCreateInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineCreateInfo.stage.module = shaderModule;
    pipelineCreateInfo.stage.pName = "main";
    pipelineCreateInfo.layout = pipelineLayout;
    VkPipeline pipeline;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline);

    // Create descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSize.descriptorCount = 2;
    VkDescriptorPoolCreateInfo descriptorPoolCreateInfo{};
    descriptorPoolCreateInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptorPoolCreateInfo.maxSets = 1;
    descriptorPoolCreateInfo.poolSizeCount = 1;
    descriptorPoolCreateInfo.pPoolSizes = &poolSize;
    VkDescriptorPool descriptorPool;
    vkCreateDescriptorPool(device, &descriptorPoolCreateInfo, nullptr, &descriptorPool);

    // Allocate descriptor set
    VkDescriptorSetAllocateInfo descriptorSetAllocInfo{};
    descriptorSetAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descriptorSetAllocInfo.descriptorPool = descriptorPool;
    descriptorSetAllocInfo.descriptorSetCount = 1;
    descriptorSetAllocInfo.pSetLayouts = &descriptorSetLayout;
    VkDescriptorSet descriptorSet;
    vkAllocateDescriptorSets(device, &descriptorSetAllocInfo, &descriptorSet);

    // Update descriptor set with buffers
    VkDescriptorBufferInfo inputBufferInfo{};
    inputBufferInfo.buffer = inputBuffer;
    inputBufferInfo.offset = 0;
    inputBufferInfo.range = bufferSize;
    VkDescriptorBufferInfo outputBufferInfo{};
    outputBufferInfo.buffer = outputBuffer;
    outputBufferInfo.offset = 0;
    outputBufferInfo.range = bufferSize;

    VkWriteDescriptorSet descriptorWrites[2] = {};
    descriptorWrites[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[0].dstSet = descriptorSet;
    descriptorWrites[0].dstBinding = 0;
    descriptorWrites[0].descriptorCount = 1;
    descriptorWrites[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[0].pBufferInfo = &inputBufferInfo;
    descriptorWrites[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptorWrites[1].dstSet = descriptorSet;
    descriptorWrites[1].dstBinding = 1;
    descriptorWrites[1].descriptorCount = 1;
    descriptorWrites[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptorWrites[1].pBufferInfo = &outputBufferInfo;
    vkUpdateDescriptorSets(device, 2, descriptorWrites, 0, nullptr);

    // Bind pipeline and descriptor set, push constants, then dispatch
    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    
    float cmyk_angles[4] = {15.0f, 75.0f, 1.0f, 45.0f};
    float cmyk_pitches[4] = {5.0f, 5.0f, 5.0f, 5.0f};
    float black_ratio = 0.05f;
    int color_levels_per_channel = 7;
    float edge_threshold = 0.3f;
    float edge_sigma = 2.0f;

    float dot_to_pixel_transforms[4][2][2];
    for (int i = 0; i < 4; i++) {
        float angle_rad = cmyk_angles[i] * 3.14159265f / 180.0f;
        float cos_a = cosf(angle_rad);
        float sin_a = sinf(angle_rad);
        dot_to_pixel_transforms[i][0][0] = cmyk_pitches[i] * cos_a; 
        dot_to_pixel_transforms[i][0][1] = -cmyk_pitches[i] * sin_a;
        dot_to_pixel_transforms[i][1][0] = cmyk_pitches[i] * sin_a;
        dot_to_pixel_transforms[i][1][1] = cmyk_pitches[i] * cos_a;
    }

    // pixel_to_dot_transforms = [x.T for x in np.linalg.inv(dot_to_pixel_transforms)]
    float pixel_to_dot_transforms[4][2][2];
    for (int i = 0; i < 4; i++) {
        float a = dot_to_pixel_transforms[i][0][0];
        float b = dot_to_pixel_transforms[i][0][1];
        float c = dot_to_pixel_transforms[i][1][0];
        float d = dot_to_pixel_transforms[i][1][1];
        float det = a * d - b * c;
        pixel_to_dot_transforms[i][0][0] =  d / det;
        pixel_to_dot_transforms[i][0][1] = -b / det;
        pixel_to_dot_transforms[i][1][0] = -c / det;
        pixel_to_dot_transforms[i][1][1] =  a / det;
    }
    
    float p2d_cyan_mat2[4] = {
        pixel_to_dot_transforms[0][0][0],
        pixel_to_dot_transforms[0][0][1],
        pixel_to_dot_transforms[0][1][0],
        pixel_to_dot_transforms[0][1][1]
    };
    float p2d_magenta_mat2[4] = {
        pixel_to_dot_transforms[1][0][0],
        pixel_to_dot_transforms[1][0][1],
        pixel_to_dot_transforms[1][1][0],
        pixel_to_dot_transforms[1][1][1]
    };
    float p2d_yellow_mat2[4] = {
        pixel_to_dot_transforms[2][0][0],
        pixel_to_dot_transforms[2][0][1],
        pixel_to_dot_transforms[2][1][0],
        pixel_to_dot_transforms[2][1][1]
    };
    float p2d_black_mat2[4] = {
        pixel_to_dot_transforms[3][0][0],
        pixel_to_dot_transforms[3][0][1],
        pixel_to_dot_transforms[3][1][0],
        pixel_to_dot_transforms[3][1][1]
    };
    float d2p_cyan_mat2[4] = {
        dot_to_pixel_transforms[0][0][0],
        dot_to_pixel_transforms[0][0][1],
        dot_to_pixel_transforms[0][1][0],
        dot_to_pixel_transforms[0][1][1]
    };
    float d2p_magenta_mat2[4] = {
        dot_to_pixel_transforms[1][0][0],
        dot_to_pixel_transforms[1][0][1],
        dot_to_pixel_transforms[1][1][0],
        dot_to_pixel_transforms[1][1][1]
    };
    float d2p_yellow_mat2[4] = {
        dot_to_pixel_transforms[2][0][0],
        dot_to_pixel_transforms[2][0][1],
        dot_to_pixel_transforms[2][1][0],
        dot_to_pixel_transforms[2][1][1]
    };
    float d2p_black_mat2[4] = {
        dot_to_pixel_transforms[3][0][0],
        dot_to_pixel_transforms[3][0][1],
        dot_to_pixel_transforms[3][1][0],
        dot_to_pixel_transforms[3][1][1]
    };

    // Push image dimensions to shader
    struct PushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t colorLevels;
        float black_ratio;
        float edge_threshold;
        float edge_sigma;
        float p2d_cyan_mat2[4];
        float p2d_magenta_mat2[4];
        float p2d_yellow_mat2[4];
        float p2d_black_mat2[4];
        float d2p_cyan_mat2[4];
        float d2p_magenta_mat2[4];
        float d2p_yellow_mat2[4];
        float d2p_black_mat2[4];
    } pushConstants;
    
    pushConstants.width = static_cast<uint32_t>(imageWidth);
    pushConstants.height = static_cast<uint32_t>(imageHeight);
    pushConstants.colorLevels = color_levels_per_channel;
    pushConstants.black_ratio = black_ratio;
    pushConstants.edge_threshold = edge_threshold;
    pushConstants.edge_sigma = edge_sigma;
    memcpy(pushConstants.p2d_cyan_mat2, p2d_cyan_mat2, sizeof(p2d_cyan_mat2));
    memcpy(pushConstants.p2d_magenta_mat2, p2d_magenta_mat2, sizeof(p2d_magenta_mat2));
    memcpy(pushConstants.p2d_yellow_mat2, p2d_yellow_mat2, sizeof(p2d_yellow_mat2));
    memcpy(pushConstants.p2d_black_mat2, p2d_black_mat2, sizeof(p2d_black_mat2));
    memcpy(pushConstants.d2p_cyan_mat2, d2p_cyan_mat2, sizeof(d2p_cyan_mat2));
    memcpy(pushConstants.d2p_magenta_mat2, d2p_magenta_mat2, sizeof(d2p_magenta_mat2));
    memcpy(pushConstants.d2p_yellow_mat2, d2p_yellow_mat2, sizeof(d2p_yellow_mat2));
    memcpy(pushConstants.d2p_black_mat2, d2p_black_mat2, sizeof(d2p_black_mat2));
    
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pushConstants), &pushConstants);
    
    vkCmdDispatch(commandBuffer, (imageWidth + 15) / 16, (imageHeight + 15) / 16, 1);


    // End recording commands
    vkEndCommandBuffer(commandBuffer);

    // Submit command buffer to the compute queue
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(computeQueue);

    
    // Save the processed image
    unsigned char* newImageData = (unsigned char*)malloc(bufferSize);
    memcpy(newImageData, output_data, bufferSize);
    stbi_write_png("output_image.png", imageWidth, imageHeight, 4, newImageData, imageWidth * 4);
    free(newImageData);
    
    // Cleanup Vulkan resources
    // TODO - anything missing?
    vkUnmapMemory(device, inputBufferMemory);
    vkUnmapMemory(device, outputBufferMemory);
    vkDestroyCommandPool(device, commandPool, nullptr);
    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);
    printf("Vulkan completed successfully.\n");

    return 0;
}