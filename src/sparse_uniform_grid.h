#include <unordered_map>
#include <vector>
#include <iostream>
#include <functional>
#include <Eigen/Core>

#pragma once

// Utility function the same as the glsl clamp
template <typename T>
inline double clamp(T x, T a, T b) {
    return std::min(std::max(x, a), b);
}

// Utility function to compute the axis aligned bounding box of the imported model
Eigen::RowVector3d bounding_box(const Eigen::MatrixXd& P) {
    // Grid bounds: axis-aligned bounding box
    Eigen::RowVector3d bb_min, bb_max;
    bb_min = P.colwise().minCoeff();
    bb_max = P.colwise().maxCoeff();

    // Bounding box dimensions
    return (bb_max - bb_min);
}



class DenseUniformGrid {
    std::vector<std::vector<size_t>> m_data;
    Eigen::MatrixXd m_points;
    double m_binWidth;
    size_t m_binsPerAxis[3];
    Eigen::RowVector3d m_origin;
    size_t m_maxPointsPerBin;
    size_t m_numPoints;

    float clamp(double n, double lower, double upper) const {
      return std::max(lower, std::min(n, upper));
    }

    #define FOR_EACH_BIN_IN_BALL(int_rad, pt) \
        for (int64_t i = -int_rad; i < int_rad+1; i += 1) { \
            for (int64_t j = -int_rad; j < int_rad+1; j += 1) { \
                for (int64_t k = -int_rad; k < int_rad+1; k += 1) { \
                    const size_t x = clamp(static_cast<size_t>(pt[0]/m_binWidth) + k, 0, m_binsPerAxis[0]-1); \
                    const size_t y = clamp(static_cast<size_t>(pt[1]/m_binWidth) + j, 0, m_binsPerAxis[1]-1); \
                    const size_t z = clamp(static_cast<size_t>(pt[2]/m_binWidth) + i, 0, m_binsPerAxis[2]-1); \
                    const size_t bin = x + m_binsPerAxis[0]*(y + z*m_binsPerAxis[1]);
    #define END_FOR_EACH_POINT_IN_BALL } } }

public:
    DenseUniformGrid(const Eigen::RowVector3d& boundingBox, const Eigen::RowVector3d& origin, double binWidth) {
        m_binWidth = binWidth;
        m_maxPointsPerBin = 0;
        m_numPoints = 0;
        for (size_t i = 0; i < 3; i += 1) {
            m_binsPerAxis[i] = static_cast<size_t>(boundingBox[i] / binWidth) + 2;
        }
        m_origin = origin - Eigen::RowVector3d(binWidth, binWidth, binWidth);

        m_data.resize(m_binsPerAxis[0]*m_binsPerAxis[1]*m_binsPerAxis[2]);
    }

    void batchInsert(const Eigen::MatrixXd& P) {
        if (m_points.rows() < (P.rows() + m_numPoints)) {
            m_points.resize(P.rows() + m_numPoints, 3);
        }

        for (size_t i = 0; i < P.rows(); i++) {
            m_points.row(m_numPoints+i) = P.row(i);
            Eigen::RowVector3d pt = P.row(i) - m_origin;
            const size_t x = static_cast<size_t>(pt[0]/m_binWidth);
            const size_t y = static_cast<size_t>(pt[1]/m_binWidth);
            const size_t z = static_cast<size_t>(pt[2]/m_binWidth);
            const size_t key = x + m_binsPerAxis[0] * (y + z*m_binsPerAxis[1]);
            m_data[key].push_back(m_numPoints+i);
            m_maxPointsPerBin = std::max(m_maxPointsPerBin, static_cast<size_t>(m_data[key].size()));
        }
        m_numPoints += P.rows();
    }

    // Find all the points in a ball of a given radius of a point. Return the index of the point in the
    // point data structure and the squared distance from the input point
    size_t pointsInBall(const Eigen::RowVector3d& p, const double radius, std::vector<std::pair<size_t, double>>& ret) const {
        const int64_t int_rad = static_cast<int64_t>(fabs(radius) / m_binWidth);
        const double rad_sq = radius*radius;

        size_t back = 0;
        Eigen::RowVector3d pt = p - m_origin;
        FOR_EACH_BIN_IN_BALL(int_rad, pt)
            for (size_t pi : m_data[bin]) {
                Eigen::RowVector3d pt = m_points.row(pi) - p;
                double sq_dist = pt.dot(pt);
                if (sq_dist < rad_sq) {
                    ret[back++] = std::make_pair<size_t, double>(static_cast<size_t>(pi), sqrt(sq_dist));
                }
            }
        END_FOR_EACH_POINT_IN_BALL
        return back;
    }

    // Finds the nearest neighbor of a point within a fixed radius of a point
    // Will crash the program if no points are found which is totally fine since
    // we're only using this for small perturbations of actual points.
    size_t nearestNeighbor(const Eigen::RowVector3d& p, double rad) const {
        size_t min_i = 0;
        double min_sq_dist = std::numeric_limits<double>::max();
        bool found = false;

        int64_t int_rad = 1 + static_cast<int64_t>(fabs(rad) / m_binWidth);

        Eigen::RowVector3d pt = p - m_origin;
        FOR_EACH_BIN_IN_BALL(int_rad, pt)
            for (size_t pi : m_data[bin]) {
                Eigen::RowVector3d pt = m_points.row(pi) - p;
                double sq_dist = pt.dot(pt);
                if (sq_dist < min_sq_dist) {
                    min_sq_dist = sq_dist;
                    min_i = pi;
                    found = true;
                }
            }
        END_FOR_EACH_POINT_IN_BALL
        assert(found);
        return min_i;
    }

    double binWidth() const {
        return m_binWidth;
    }

    size_t maxPointsPerBin() const {
        return m_maxPointsPerBin;
    }

    size_t size() const {
        return m_numPoints;
    }

    const Eigen::MatrixXd& points() const {
        return m_points;
    }
};


//// A uniform grid used as an acceleration structure for nearest neighbor and ball queries
//class SparseUniformGrid {

//    // This code copied from boost::hash_combine which lets me combine multiple hash functions
//    // into a single hash function.
//    struct Point3dHasher {
//        std::size_t operator()(const std::tuple<int64_t, int64_t, int64_t>& p) const {
//            std::hash<int64_t> hasher;
//            size_t ret = 0;
//            ret ^= hasher(std::get<0>(p)) + 0x9e3779b9 + (ret<<6) + (ret>>2);
//            ret ^= hasher(std::get<1>(p)) + 0x9e3779b9 + (ret<<6) + (ret>>2);
//            ret ^= hasher(std::get<2>(p)) + 0x9e3779b9 + (ret<<6) + (ret>>2);
//            return ret;
//        }
//    };

//    typedef std::tuple<int64_t, int64_t, int64_t> PointTuple;
//    typedef std::unordered_map<PointTuple, std::vector<size_t>, Point3dHasher> Point3dHashTable;

//    PointTuple getKey(const Eigen::RowVector3d& p) const {
//        return std::make_tuple(static_cast<int64_t>(p[0] / m_binWidth),
//                               static_cast<int64_t>(p[1] / m_binWidth),
//                               static_cast<int64_t>(p[2] / m_binWidth));
//    }

//    PointTuple getKey(const PointTuple& p, int64_t offsetX, int64_t offsetY, int64_t offsetZ) const {
//        return std::make_tuple(offsetX + std::get<0>(p),
//                               offsetY + std::get<1>(p),
//                               offsetZ + std::get<2>(p));
//    }

//    Eigen::MatrixXd m_points;
//    double m_binWidth;
//    size_t m_numPoints;
//    size_t m_maxPointsPerBin;
//    Point3dHashTable m_sparseGrid;

//#define FOR_EACH_BIN_IN_BALL(int_rad, p) \
//    PointTuple ctr = getKey(p); \
//    for (int64_t i = -int_rad; i < int_rad+1; i += 1) { \
//        for (int64_t j = -int_rad; j < int_rad+1; j += 1) { \
//            for (int64_t k = -int_rad; k < int_rad+1; k += 1) { \
//                PointTuple bin = getKey(ctr, i, j, k); \
//                auto binval = m_sparseGrid.find(bin);\
//                if (binval != m_sparseGrid.end()) {
//#define END_FOR_EACH_POINT_IN_BALL } } } }

//public:
//    SparseUniformGrid(double binWidth) : m_binWidth(binWidth), m_maxPointsPerBin(0) {}

//    SparseUniformGrid(size_t size, double binWidth) :
//        m_binWidth(binWidth), m_maxPointsPerBin(0), m_sparseGrid(Point3dHashTable(2*size)), m_numPoints(0) {
//        m_points = Eigen::MatrixXd(size, 3);
//    }

//    void clear() {
//        m_maxPointsPerBin = 0;
//        m_numPoints = 0;
//        m_points = Eigen::MatrixXd();
//        m_sparseGrid.clear();
//    }

//    void batchInsert(const Eigen::MatrixXd& P) {
//        if (m_points.rows() < (P.rows() + m_numPoints)) {
//            m_points.resize(P.rows() + m_numPoints, 3);
//        }

//        for (size_t i = 0; i < P.rows(); i++) {
//            m_points.row(m_numPoints+i) = P.row(i);
//            auto key = getKey(P.row(i));
//            if (m_sparseGrid.find(key) == m_sparseGrid.end()) {
//                m_sparseGrid[key] = std::vector<size_t>();
//            }
//            m_sparseGrid.at(key).push_back(m_numPoints+i);
//            m_maxPointsPerBin = std::max(m_maxPointsPerBin, static_cast<size_t>(m_sparseGrid.at(key).size()));
//        }
//        m_numPoints += P.rows();
//    }

//    void insert(const Eigen::RowVector3d& P) {
//        if (m_points.rows() < (1 + m_numPoints)) {
//            m_points.resize(1 + m_numPoints, 3);
//        }

//        m_points.row(m_numPoints) = P;
//        auto key = getKey(P);
//        if (m_sparseGrid.find(key) == m_sparseGrid.end()) {
//            m_sparseGrid[key] = std::vector<size_t>();
//        }
//        m_sparseGrid.at(key).push_back(m_numPoints);
//        m_maxPointsPerBin = std::max(m_maxPointsPerBin, static_cast<size_t>(m_sparseGrid.at(key).size()));
//        m_numPoints += 1;
//    }

//    // Find all the points in a ball of a given radius of a point. Return the index of the point in the
//    // point data structure and the squared distance from the input point
//    size_t pointsInBall(const Eigen::RowVector3d& p, const double radius, std::vector<std::pair<size_t, double>>& ret) const {
//        const int64_t int_rad = static_cast<int64_t>(fabs(radius) / m_binWidth);
//        const double rad_sq = radius*radius;

//        size_t back = 0;
//        FOR_EACH_BIN_IN_BALL(int_rad, p)
//            for (size_t pi : binval->second) {
//                Eigen::RowVector3d pt = m_points.row(pi) - p;
//                double sq_dist = pt.dot(pt);
//                if (sq_dist < rad_sq) {
//                    ret[back++] = std::make_pair<size_t, double>(static_cast<size_t>(pi), sqrt(sq_dist));
//                }
//            }
//        END_FOR_EACH_POINT_IN_BALL
//        return ret.size();
//    }

//    // Finds the nearest neighbor of a point within a fixed radius of a point
//    // Will crash the program if no points are found which is totally fine since
//    // we're only using this for small perturbations of actual points.
//    size_t nearestNeighbor(const Eigen::RowVector3d& p, double rad) const {
//        size_t min_i = 0;
//        double min_sq_dist = std::numeric_limits<double>::max();
//        bool found = false;

//        int64_t int_rad = 1 + static_cast<int64_t>(fabs(rad) / m_binWidth);

//        FOR_EACH_BIN_IN_BALL(int_rad, p)
//            for (int64_t pi : m_sparseGrid.at(bin)) {
//                auto pt = m_points.row(pi) - p;
//                double sq_dist = pt.dot(pt);
//                if (sq_dist < min_sq_dist) {
//                    min_sq_dist = sq_dist;
//                    min_i = pi;
//                    found = true;
//                }
//            }
//        END_FOR_EACH_POINT_IN_BALL
//        assert(found);
//        return min_i;
//    }

//    double binWidth() const {
//        return m_binWidth;
//    }

//    size_t maxPointsPerBin() const {
//        return m_maxPointsPerBin;
//    }

//    size_t size() const {
//        return m_numPoints;
//    }

//    const Eigen::MatrixXd& points() const {
//        return m_points;
//    }
//};

