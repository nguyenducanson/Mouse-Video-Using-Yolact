import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

def get_edges(polygon):
    """
    :param polygon: a list of points (point = list or tuple holding two numbers)
    :return: the edges of the polygon, i.e. all pairs of points
    """
    for i in range(len(polygon)):
        yield Edge(polygon[i - 1], polygon[i])


class Edge:
    def __init__(self, point_a, point_b):
        self._support_vector = np.array(point_a)
        self._direction_vector = np.subtract(point_b, point_a)

    def get_intersection_point(self, other):
        t = self._get_intersection_parameter(other)
        return None if t is None else self._get_point(t)

    def _get_point(self, parameter):
        return self._support_vector + parameter * self._direction_vector

    def _get_intersection_parameter(self, other):
        A = np.array([-self._direction_vector, other._direction_vector]).T
        if np.linalg.matrix_rank(A) < 2:
            return None
        b = np.subtract(self._support_vector, other._support_vector)
        x = np.linalg.solve(A, b)
        return x[0] if 0 <= x[0] <= 1 and 0 <= x[1] <= 1 else None


def intersect(polygon1, polygon2):
    """
    The given polygons must be convex and their vertices must be in anti-clockwise order (this is not checked!)

    Example: polygon1 = [[0,0], [0,1], [1,1]]

    """
    polygon3 = list()
    polygon3 += _get_vertices_lying_in_the_other_polygon(polygon1, polygon2)
    polygon3 += _get_edge_intersection_points(polygon1, polygon2)
    return _sort_vertices_anti_clockwise_and_remove_duplicates(polygon3)


def _get_vertices_lying_in_the_other_polygon(polygon1, polygon2):
    vertices = list()
    vertices += [vertex for vertex in polygon1 if _polygon_contains_point(polygon2, vertex)]
    vertices += [vertex for vertex in polygon2 if _polygon_contains_point(polygon1, vertex)]
    return vertices


def _get_edge_intersection_points(polygon1, polygon2):
    intersection_points = list()
    for edge1 in get_edges(polygon1):
        for edge2 in get_edges(polygon2):
            intersection_point = edge1.get_intersection_point(edge2)
            if intersection_point is not None:
                intersection_points.append(intersection_point)
    return intersection_points


def _polygon_contains_point(polygon, point):
    for i in range(len(polygon)):
        a = np.subtract(polygon[i], polygon[i - 1])
        b = np.subtract(point, polygon[i - 1])
        if np.cross(a, b) < 0:
            return False
    return True


def _sort_vertices_anti_clockwise_and_remove_duplicates(polygon, tolerance=1e-7):
    polygon = sorted(polygon, key=lambda p: _get_angle_in_radians(_get_bounding_box_midpoint(polygon), p))

    def vertex_not_similar_to_previous(_polygon, i):
        diff = np.subtract(_polygon[i - 1], _polygon[i])
        return np.linalg.norm(diff, np.inf) > tolerance

    return [p for i, p in enumerate(polygon) if vertex_not_similar_to_previous(polygon, i)]


def _get_angle_in_radians(point1, point2):
    return np.arctan2(point2[1] - point1[1], point2[0] - point1[0])


def _get_bounding_box_midpoint(polygon):
    x = [p[0] for p in polygon]
    y = [p[1] for p in polygon]
    return [(np.max(x) + np.min(x)) / 2., (np.max(y) + np.min(y)) / 2.]


def visualize_polygon():
    def generate_random_convex_polygon():
          return _sort_vertices_anti_clockwise_and_remove_duplicates(
              [[np.cos(x), np.sin(x)] for x in np.random.rand(random.randint(3, 6)) * 2 * np.pi])


    def plot_polygon(polygon):
        if polygon:
            _polygon = list(polygon)
            _polygon.append(_polygon[0])
            x, y = zip(*_polygon)
            plt.plot(x, y, 'o-')
            plt.fill(x, y, alpha=0.25)


    polygon1 = generate_random_convex_polygon()
    polygon2 = generate_random_convex_polygon()
    polygon3 = intersect(polygon1, polygon2)

    plot_polygon(polygon1)
    plot_polygon(polygon2)
    plot_polygon(polygon3)
    print(type(polygon3), type(polygon3[0]))
    plt.savefig('convex.png')


def generate_convex_polygon(point_list):
    curr_set_of_point = np.array(point_list)
    curr_hull = ConvexHull(curr_set_of_point)
    result = []
    for index in curr_hull.vertices:
        result.append(point_list[index])
    return result


def calc_polygon_area(vertices):
    if len(vertices) < 3:
      return 0
    psum = 0
    nsum = 0

    for i in range(len(vertices)):
        sindex = (i + 1) % len(vertices)
        prod = vertices[i][0] * vertices[sindex][1]
        psum += prod

    for i in range(len(vertices)):
        sindex = (i + 1) % len(vertices)
        prod = vertices[sindex][0] * vertices[i][1]
        nsum += prod

    return abs(1/2*(psum - nsum))


def calculate_intersection_area(mouse_points, port_points):
    mouse_polygon = generate_convex_polygon(mouse_points)
    port_polygon = generate_convex_polygon(port_points)

    mouse_area = calc_polygon_area(mouse_polygon)

    intersect_polygon = intersect(mouse_polygon, port_polygon)
    return calc_polygon_area(intersect_polygon) / mouse_area

