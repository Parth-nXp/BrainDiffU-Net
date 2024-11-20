def split_data_into_nodes(image_filenames, mask_images, num_nodes):
    """
    Split image and mask file paths into chunks for each node.

    Parameters:
    image_filenames (list): List of file paths for images.
    mask_images (list): List of file paths for masks.
    num_nodes (int): Number of nodes to split data into.

    Returns:
    tuple: Two lists of lists, each containing file paths for images and masks per node.
    """
    print(f"Splitting data into {num_nodes} nodes.")
    chunk_size = len(image_filenames) // num_nodes
    chunks_images = [image_filenames[i * chunk_size:(i + 1) * chunk_size] for i in range(num_nodes)]
    chunks_masks = [mask_images[i * chunk_size:(i + 1) * chunk_size] for i in range(num_nodes)]
    leftover = len(image_filenames) % num_nodes
    if leftover > 0:
        chunks_images[-1].extend(image_filenames[-leftover:])
        chunks_masks[-1].extend(mask_images[-leftover:])
    return chunks_images, chunks_masks
