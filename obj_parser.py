
def parse_obj(filename):
    f = open(filename)
    lines = f.readlines()
    vertices = []
    objects = []

    for line in lines:
        words = line.split()
        if len(words) == 0:
            continue
        if words[0] == "v":
            vertices.append(list(map(float, words[1:])))
        if words[0] == "f":
            obj = []
            for vertex_index in words[1:]:
                idx = int(vertex_index.split("/")[0]) - 1
                obj.append(
                    list(vertices[idx])
                )
            objects.append(obj)

    for object in objects:
        for point in object:
            point.append(1)
    return objects
