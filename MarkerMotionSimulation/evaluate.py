def fill_zeros(image):
    """
    Use linear interpolation to fill zeros

    @param image: (H, W) array representing one channel of the image
    @return: filled: (H, W) array with zeros filled
    """
    points = np.nonzero(image)
    values = image[points].ravel()
    xi = np.meshgrid(np.arange(0, image.shape[0]), np.arange(0, image.shape[1]))

    # for some reason we need the transpose here
    filled = interpolate.griddata(
        points, values, tuple(xi), method="linear", fill_value=0
    ).T

    return filled

if __name__ == "__main__":
    obj = "0630_dome_star_0.2_0.4_0.4"
    path = "../data/FEM/" + obj + "/" + obj + "_"
    x_path = path + "x.txt"
    y_path = path + "y.txt"
    z_path = path + "z.txt"
    dl = DataLoader(x_path, y_path, z_path)
    z_map, dx_map, dy_map, dz_map = dl.generate_gt()

    h = H // 2
    w = W // 2
    d = D // 2
    crop = lambda M: M[d - h:d + h, d - w:d + w]

    # visualize
    plt.figure(0)

    plt.subplot(4, 1, 1)
    fig = plt.imshow(fill_zeros(crop(z_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 2)
    fig = plt.imshow(fill_zeros(crop(dx_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 3)
    fig = plt.imshow(fill_zeros(crop(dy_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(4, 1, 4)
    fig = plt.imshow(fill_zeros(crop(dz_map)), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-obj", default="square",
                        help="Test object; supported: square, cylinder6")
    parser.add_argument("-dx", default=0.0, type=float,
                        help="Load on the x axis")
    parser.add_argument("-dy", default=0.0, type=float,
                        help="Load on the y axis")
    parser.add_argument("-dz", default=1.0, type=float,
                        help="Load on the z axis")
    args = parser.parse_args()

    # obtain contact mask and gel map
    object_path = os.path.join("..", "data", "objects", "%s.ply" % args.obj)
    dome_map = np.load(os.path.join("..", "calibs", "dome_gel.npy"))
    raw_deform = np.array([args.dx, args.dy, args.dz])
    press_depth = raw_deform[2]
    contact_mask, gel_map = press_object(object_path, dome_map, press_depth)

    # obtain result map
    fem_path = os.path.join("..", "calibs", "femCalib.npz")
    sp = Superposition(fem_path)
    result_map = sp.propagate_deform(raw_deform, contact_mask, gel_map)

    # TODO: should we crop before interpolation or after?

    # crop by taking a (H, W) subarray at the center of the (D, D) array
    h = H // 2
    w = W // 2
    d = D // 2
    cropped_map = result_map[d - h:d + h, d - w:d + w, :]

    # visualize
    plt.figure(0)

    plt.subplot(3, 1, 1)
    fig = plt.imshow(fill_zeros(cropped_map[:, :, 0]), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 1, 2)
    fig = plt.imshow(fill_zeros(cropped_map[:, :, 1]), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.subplot(3, 1, 3)
    fig = plt.imshow(fill_zeros(cropped_map[:, :, 2]), cmap="RdBu")
    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.show()
    # output_path = os.path.join("..", "results", "%s_compose.jpg" % args.obj)
    # plt.savefig(output_path)
