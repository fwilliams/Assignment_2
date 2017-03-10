#include <memory>
#include <igl/readOFF.h>
#include <igl/viewer/Viewer.h>
/*** insert any necessary libigl headers here ***/
#include <igl/per_face_normals.h>
#include <igl/copyleft/marching_cubes.h>
#include "sparse_uniform_grid.h"


using namespace std;
using Viewer = igl::viewer::Viewer;

// Input: imported points, #P x3
Eigen::MatrixXd P;

// Input: imported normals, #P x3
Eigen::MatrixXd N;

// Parameter: degree of the polynomial
unsigned int polyDegree = 0;

// Parameter: Wendland weight function radius (make this relative to the size of the mesh)
double wendlandRadius = 0.1;

// Parameter: grid resolution
unsigned int resolution = 20;

// Parameter: the size of the points to render
unsigned pointSize = 4;

// Intermediate result: constrained points, #C x3
Eigen::MatrixXd constrainedPoints;

// Intermediate result: implicit function values at constrained points, #C x1
Eigen::VectorXd constrainedValues;

// Intermediate result: grid points, at which the imlicit function will be evaluated, #G x3
Eigen::MatrixXd grid_points;

// Intermediate result: implicit function values at the grid points, #G x1
Eigen::VectorXd grid_values;

// Intermediate result: grid point colors, for display, #G x3
Eigen::MatrixXd grid_colors;

// Intermediate result: grid lines, for display, #L x6 (each row contains
// starting and ending point of line segment)
Eigen::MatrixXd grid_lines;

// Output: vertex array, #V x3
Eigen::MatrixXd V;

// Output: face array, #F x3
Eigen::MatrixXi F;

// Output: face normals of the reconstructed mesh, #F x3
Eigen::MatrixXd FN;

// Data structure to accelerate nearest neighbor and ball queries
std::unique_ptr<DenseUniformGrid> uniform_grid;

// Functions
void createGrid();
void evaluateImplicitFunc();
void getLines();
bool callbackKeyDown(Viewer& viewer, unsigned char key, int modifiers);

Eigen::RowVector3d boundingBox;

// Count the number of monomials in a polynomial of
size_t numMonomials(size_t deg) {
    size_t ret = 0;

    const size_t degree = deg + 1;
    for (size_t i = 0; i < degree; i += 1) {
        for (size_t j = 0; j < (degree - i); j += 1) {
            for (size_t k = 0; k < (degree - i - j); k += 1) {
                ret += 1;
            }
        }
    }
    return ret;
}

inline double fastPow(double a, size_t b) {
  double res = 1.0;
  for(size_t i = 0; i < b; i++) {
      res *= a;
  }
  return res;
}

Eigen::RowVectorXd enumerateMonomials(const Eigen::RowVector3d& point, size_t outsize) {
    size_t pd1 = polyDegree + 1;

    Eigen::RowVectorXd ret(outsize);
    size_t back = 0;
    for (size_t i = 0; i < pd1; i += 1) {
        for (size_t j = 0; j < (pd1 - i); j += 1) {
            for (size_t k = 0; k < (pd1 - i - j); k += 1) {
                ret(back) = fastPow(point(0), i) * fastPow(point(1), j) * fastPow(point(2), k);
                back += 1;
            }
        }
    }
    return ret;
}

double evalPolynomial(const Eigen::RowVectorXd& coeffs, const Eigen::Vector3d& point) {
    size_t pd1 = polyDegree + 1;

    double ret = 0.0;
    size_t back = 0;
    for (size_t i = 0; i < pd1; i += 1) {
        for (size_t j = 0; j < (pd1 - i); j += 1) {
            for (size_t k = 0; k < (pd1 - i - j); k += 1) {
                ret += coeffs(back) * fastPow(point(0), i) * fastPow(point(1), j) * fastPow(point(2), k);
                back += 1;
            }
        }
    }
    return ret;
}

// Brute force nearest neighbor search
size_t nearest_neighbor_brute_force(const Eigen::RowVector3d& p) {
    size_t min_i = 0;
    double min_dist = std::numeric_limits<double>::max();
    for (size_t i = 0; i < P.rows(); i += 1) {
        double dist = (P.row(i) - p).norm();
        if (dist < min_dist) {
            min_dist = dist;
            min_i = i;
        }
    }
    return min_i;
}


// Compute the set of constraints used to solve the implicit function
void computeConstraints() {
    Eigen::RowVector3d boundingBox = bounding_box(P);
    Eigen::RowVector3d origin = P.colwise().minCoeff();
    double binWidth = boundingBox.norm() / pow(P.rows()*3, 1.0/3.0);
    uniform_grid = std::unique_ptr<DenseUniformGrid>(new DenseUniformGrid(boundingBox*1.1, origin-0.05*boundingBox, binWidth));

    cout << "Generating constraints..." << endl;
    uniform_grid->batchInsert(P);

    constrainedPoints.resize(3*P.rows(), 3);
    constrainedValues.resize(3*P.rows());
    const double epsilon = bounding_box(P).norm() * 0.01;

    for (size_t i = 0; i < P.rows(); i += 1) {
        N.row(i) = N.row(i) / N.row(i).norm();
        constrainedPoints.row(i) = P.row(i);
        constrainedValues[i] = 0.0;

        Eigen::RowVector3d p_c = P.row(i) + N.row(i) * epsilon;
        for (double j = 0.5; uniform_grid->nearestNeighbor(p_c, epsilon) != i; j *= 0.5) {
            p_c = P.row(i) + N.row(i) * (epsilon * j);
        }
        constrainedPoints.row(P.rows()+i) = p_c;
        constrainedValues[P.rows()+i] = (p_c - P.row(i)).norm();

        p_c = P.row(i) - (N.row(i) * epsilon);
        for (double j = 0.5; uniform_grid->nearestNeighbor(p_c, epsilon) != i; j *= 0.5) {
            p_c = P.row(i) - (N.row(i) * (epsilon * j));
        }
        constrainedPoints.row(2*P.rows()+i) = p_c;
        constrainedValues[2*P.rows()+i] = -(p_c - P.row(i)).norm();
    }
    uniform_grid->batchInsert(constrainedPoints.block(P.rows(), 0, 2*P.rows(), 3));
    cout << "Uniform grid has at most " << uniform_grid->maxPointsPerBin() << " points per bin" << std::endl;
}

// Creates a grid_points array for the simple sphere example. The points are
// stacked into a single matrix, ordered first in the x, then in the y and
// then in the z direction. If you find it necessary, replace this with your own
// function for creating the grid.
void createGrid() {
    grid_points.resize(0, 3);
    grid_colors.resize(0, 3);
    grid_lines. resize(0, 6);
    grid_values.resize(0);
    V. resize(0, 3);
    F. resize(0, 3);
    FN.resize(0, 3);

    Eigen::RowVector3d mean = P.colwise().mean();
    Eigen::MatrixXd PP = P.rowwise()-mean;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(PP.transpose() * PP);
    Eigen::MatrixXd rot = solver.eigenvectors();
    if (solver.eigenvectors().determinant() < 0.0) {
        rot.col(0) = solver.eigenvectors().col(1);
        rot.col(1) = solver.eigenvectors().col(0);
    }

    // Grid bounds: axis-aligned bounding box
    Eigen::MatrixXd Prot(P.rows(), 3);
    for(size_t i = 0; i < P.rows(); i++) {
        Prot.row(i) = (rot.transpose() * PP.row(i).transpose()).transpose();
    }
    Eigen::RowVector3d bb_min = Prot.colwise().minCoeff() + mean;
    Eigen::RowVector3d bb_max = Prot.colwise().maxCoeff() + mean;
    boundingBox = bb_max-bb_min;
    Eigen::RowVector3d dim = (bb_max - bb_min)*1.1;

    // Grid spacing
    const double dx = dim[0] / (double)(resolution - 1);
    const double dy = dim[1] / (double)(resolution - 1);
    const double dz = dim[2] / (double)(resolution - 1);

    // 3D positions of the grid points -- see slides or marching_cubes.h for ordering
    grid_points.resize(resolution * resolution * resolution, 3);

    // Create each gridpoint
    for (unsigned int x = 0; x < resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                // Linear index of the point at (x,y,z)
                Eigen::RowVector3d pt = bb_min -0.05*(boundingBox) + Eigen::RowVector3d(x*dx, y*dy, z*dz) - mean;
                Eigen::Vector3d p = rot * pt.transpose() + mean.transpose();
                int index = x + resolution * (y + resolution * z);
                // 3D point at (x,y,z)
                grid_points.row(index) = p.transpose();
            }
        }
    }
}

// Function for explicitly evaluating the implicit function for a sphere of
// radius r centered at c : f(p) = ||p-c|| - r, where p = (x,y,z).
// This will NOT produce valid results for any mesh other than the given
// sphere.
// Replace this with your own function for evaluating the implicit function
// values at the grid points using MLS
void evaluateImplicitFunc() {
    grid_values.resize(grid_points.rows());
    double wr = boundingBox.norm()*wendlandRadius;

    cout << "There are " << grid_points.rows() << " grid points" << endl;
    cout << "The wendland radius is " << wr << endl;

    const size_t int_rad = static_cast<size_t>(fabs(wr) / uniform_grid->binWidth())+1;
    const size_t n = numMonomials(polyDegree);
    std::vector<std::pair<size_t, double>> ptsInRad(int_rad*int_rad*int_rad*uniform_grid->maxPointsPerBin());
    for (size_t i = 0; i < grid_points.rows(); i+= 1) {
        size_t numPtsInRad = uniform_grid->pointsInBall(grid_points.row(i), wr, ptsInRad);
        if (numPtsInRad < n) {
            grid_values[i] = INFINITY;
            continue;
        }

        Eigen::MatrixXd B(numPtsInRad, n);
        Eigen::DiagonalMatrix<double, Eigen::Dynamic> W(numPtsInRad);
        Eigen::VectorXd d(numPtsInRad);
        for(size_t j = 0; j < numPtsInRad; j += 1) {
            double rOverH = ptsInRad[j].second / wr;
            W.diagonal()[j] = pow(1.0 - rOverH, 4.0) * (4.0*rOverH + 1.0);
            B.row(j) = enumerateMonomials(constrainedPoints.row(ptsInRad[j].first), n);
            d[j] = constrainedValues[ptsInRad[j].first];
        }

        Eigen::MatrixXd B_t_W = B.transpose() * W;
        Eigen::MatrixXd A = B_t_W * B;
        Eigen::MatrixXd b = B_t_W * d;
        Eigen::VectorXd sol = A.colPivHouseholderQr().solve(b);
//        Eigen::VectorXd sol = (W*B).jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(d);
        grid_values[i] = evalPolynomial(sol, grid_points.row(i));
    }

    cout << "Done!" << endl;
}

// Code to display the grid lines given a grid structure of the given form.
// Assumes grid_points have been correctly assigned
// Replace with your own code for displaying lines if need be.
void getLines() {
    int nnodes = grid_points.rows();
    grid_lines.resize(3 * nnodes, 6);
    int numLines = 0;

    for (unsigned int x = 0; x<resolution; ++x) {
        for (unsigned int y = 0; y < resolution; ++y) {
            for (unsigned int z = 0; z < resolution; ++z) {
                int index = x + resolution * (y + resolution * z);
                if (x < resolution - 1) {
                    int index1 = (x + 1) + y * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (y < resolution - 1) {
                    int index1 = x + (y + 1) * resolution + z * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
                if (z < resolution - 1) {
                    int index1 = x + y * resolution + (z + 1) * resolution * resolution;
                    grid_lines.row(numLines++) << grid_points.row(index), grid_points.row(index1);
                }
            }
        }
    }

    grid_lines.conservativeResize(numLines, Eigen::NoChange);
}



bool callbackKeyDown(Viewer &viewer, unsigned char key, int modifiers) {
    if (key == '1') {
        // Show imported points
        viewer.data.clear();
        viewer.core.align_camera_center(P);
        viewer.core.point_size = 11;
        viewer.data.add_points(P, Eigen::RowVector3d(0,0,0));
    }

    if (key == '2') {
        // Show all constraints
        viewer.data.clear();
        viewer.core.align_camera_center(P);

        computeConstraints();

        viewer.core.point_size = pointSize;
        viewer.data.add_points(constrainedPoints.block(P.rows(), 0, P.rows(), 3), Eigen::RowVector3d(0.0, 1.0, 0.0));
        viewer.data.add_points(constrainedPoints.block(2*P.rows(), 0, P.rows(), 3), Eigen::RowVector3d(0.0, 0.0, 1.0));
        viewer.data.add_points(P, Eigen::RowVector3d(0.0, 0.0, 0.0));
    }

    if (key == '3') {
        // Show grid points with colored nodes and connected with lines
        viewer.data.clear();
        viewer.core.align_camera_center(P);

        // Make grid
        createGrid();

        // Evaluate implicit function
        evaluateImplicitFunc();

        // get grid lines
        getLines();

        // Code for coloring and displaying the grid points and lines
        // Assumes that grid_values and grid_points have been correctly assigned.
        grid_colors.setZero(grid_points.rows(), 3);

        // Build color map
        for (int i = 0; i < grid_points.rows(); ++i) {
            double value = grid_values(i);
            if (value < 0) {
                grid_colors(i, 1) = 1;
            } else if (value > 0) {
                grid_colors(i, 0) = 1;
            }
        }

        // Draw lines and points
        viewer.core.point_size = 8;
        viewer.data.add_points(grid_points, grid_colors);
        viewer.data.add_edges(grid_lines.block(0, 0, grid_lines.rows(), 3),
                              grid_lines.block(0, 3, grid_lines.rows(), 3),
                              Eigen::RowVector3d(0.8, 0.8, 0.8));
    }

    if (key == '4') {
        // Show reconstructed mesh
        viewer.data.clear();
        // Code for computing the mesh (V,F) from grid_points and grid_values
        if ((grid_points.rows() == 0) || (grid_values.rows() == 0)) {
            cerr << "Not enough data for Marching Cubes !" << endl;
            return true;
        }
        // Run marching cubes
        igl::copyleft::marching_cubes(grid_values, grid_points, resolution, resolution, resolution, V, F);
        if (V.rows() == 0) {
            cerr << "Marching Cubes failed!" << endl;
            return true;
        }

        igl::per_face_normals(V, F, FN);
        viewer.data.set_mesh(V, F);
        viewer.core.show_lines = true;
        viewer.core.show_faces = true;
        viewer.data.set_normals(FN);
    }

    return true;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        cout << "Usage ex2_bin mesh.off" << endl;
        exit(0);
    }

    // Read points and normals
    igl::readOFF(argv[1],P,F,N);

    Viewer viewer;
    viewer.callback_key_down = callbackKeyDown;

    viewer.callback_init = [&](Viewer &v) {
        // Add widgets to the sidebar.
        v.ngui->addGroup("Reconstruction Options");
        v.ngui->addVariable("Resolution", resolution);
        v.ngui->addVariable("Wendland Radius", wendlandRadius);
        v.ngui->addVariable("Polynomial Degree", polyDegree);
        v.ngui->addButton("Reset Grid", [&](){
            // Recreate the grid
            createGrid();
            // Switch view to show the grid
            callbackKeyDown(v, '3', 0);
        });

        v.screen->performLayout();
        return false;
    };

    viewer.launch();
}
