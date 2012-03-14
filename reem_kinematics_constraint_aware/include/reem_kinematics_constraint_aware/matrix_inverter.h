/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2011, PAL Robotics, S.L.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of PAL Robotics, S.L. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 */

/** \author Adolfo Rodriguez Tsouroukdissian. */

// BIG FAT NOTE: This file was initially intended to be part of a patchset for the KDL, but my views on how to implement IK
// building blocks have changed quite a bit since the initial implementation. However, the functionality provided by
// this file is still useful, and until I get the time to refactor this code, I'm stuck with it.

#ifndef KDL_MATRIXINVERTER_H
#define KDL_MATRIXINVERTER_H

// C++ standard headers
#include <algorithm>
#include <cassert>

// Boost headers
#include <boost/shared_ptr.hpp>

// Eigen headers
#include <Eigen/Dense>

// Forward declarations
namespace Eigen
{
namespace internal
{
template<typename MatrixType> class LsInverseReturnType;
template<typename MatrixType> class DlsInverseReturnType;
template<typename MatrixType> class LsSolveReturnType;
template<typename MatrixType> class DlsSolveReturnType;
} // internal
} // Eigen

namespace KDL
{

/// \brief \e Least-squares and \e damped \e least-squares inverse computation for rectangular, real-valued matrices.
/// \tparam MatrixType Type of underlying dense matrix.
/// This class uses the Singular Value Decomposition (SVD) for computing the inverse of a matrix.
/// Let \f$ A = U S V^T \f$ be the SVD of a \e n-by-p, real-valued matrix, and let \e m be the smaller
/// value among \e n and \e p, its inverse can be computed as \f$ A^+ = V S^+ U^T \f$.
/// This class allows two ways of computing \f$ S^+ \f$: TODO: Finish
///   - <b>Least squares:</b> Yields the least-squares solution. To compute it, every nonzero entry in \e S is
///   replaced by its reciprocal. A \e fuzzy comparison citerion is used to determine the nonzero entries (see lsInverse()).
///   - <b> Damped least-squares:</b> Realizes a tradeoff between the least-squares and minimum-norm properties.
///
/// It's important to note that a single SVD can be used to compute \e both the \e least-squares and \e damped
/// \e least-squares inverses of a matrix. The following code exemplifies typical usage scenarios:
/// \code
/// typedef Eigen::MatrixXd Matrix;
/// typedef boost::shared_ptr<Matrix> MatrixPtr;
/// typedef Eigen::VectorXd Vector;
///
/// const int rows = 5;
/// const int cols = 10;
/// Matrix A = Matrix::Random(rows, cols);
/// Matrix Ainv(rows, cols);
///
/// // Different use cases of increasing complexity
/// // Case 1: Use default parameters, compute least-squares inverse
/// Ainv = MatrixInverter<Matrix>(A).lsInverse(); // Handy for one-off usage
/// Ainv = MatrixInverter<Matrix>(A).inverse();   // Same as above, inverse() is a convenience alias for lsInverse()
///
/// // Case 2: similar to case 1, but with resource preallocation
/// KDL::MatrixInverter<Matrix> inv(rows, cols); // Preallocate resources
/// Ainv = inv.compute(A).inverse();             // No allocations take place here
///
/// // Case 3: Use custom parameters
/// inv.setLsInverseThreshold(1e-5);
/// Ainv = inv.compute(A).lsInverse();
///
/// // Case 4: Compute both least-squares amd damped least-squares inverses
/// inv.setDlsInverseThreshold(0.1); // Damping activation threshold
/// inv.compute(A);                  // Using the same SVD...
/// Ainv         = inv.lsInverse();  // Compute the least-squares...
/// Matrix Adinv = inv.dlsInverse(); // and damped least-squares inverses
///
/// \endcode

// TODO: A note on concurrency

// TODO: Doc solve() methods, remove weights from example

// TODO: Incorporate into JacobiSVD?

// TODO: Add isInitialized


template <typename MatrixType>
class MatrixInverter
{
public:
  typedef MatrixType Matrix;
  typedef boost::shared_ptr<Matrix> MatrixPtr;
  typedef typename Eigen::internal::plain_col_type<Matrix>::type Vector;
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;
  typedef typename Eigen::JacobiSVD<Matrix, Eigen::ColPivHouseholderQRPreconditioner> SVD;
  typedef Eigen::internal::LsInverseReturnType<Matrix> LsInverseReturnType;
  typedef Eigen::internal::DlsInverseReturnType<Matrix> DlsInverseReturnType;
  typedef Eigen::internal::LsSolveReturnType<Matrix> LsSolveReturnType;
  typedef Eigen::internal::DlsSolveReturnType<Matrix> DlsSolveReturnType;

  /// \brief Problem dimension constructor.
  /// Preallocates all data structures so that compute(const Matrix&), lsInverse() and dlsInverse() are
  /// heap-allocation-free.
  MatrixInverter(const Index rows, const Index cols);

  /// \brief Input matrix constructor.
  /// Using this constructor, like so
  /// \code
  /// Matrix A(rows, cols);
  /// MatrixInverter<Matrix> inv(A);
  /// \endcode
  /// is a more compact and equivalent way of doing
  /// \code
  /// Matrix A(rows, cols);
  /// MatrixInverter<Matrix> inv(rows, cols);
  /// inv.compute();
  /// \endcode
  /// \sa MatrixInverter(const Index rows, const Index cols)
  MatrixInverter(const Matrix& A);

  /// \param A Matrix to invert.
  /// \return A reference to \e this.
  MatrixInverter& compute(const Matrix& A);

  /// \brief Convenience alias for lsInverse(const Vector&)
  /// \sa lsInverse(const Vector&)
  LsInverseReturnType inverse() {return lsInverse();}

  /// \brief Compute the \e least-squares inverse.
  /// \return Expression object representing the \e least-squares inverse.
  /// \pre Must be called after compute(const Matrix& A).
  /// \sa setLsInverseThreshold(const Scalar eps), getLsInverseThreshold()
  LsInverseReturnType lsInverse() {return LsInverseReturnType(*this);}

  /// \brief Compute the \e damped \e least-squares inverse.
  /// \return Expression object representing the \e damped \e least-squares inverse.
  /// \pre Must be called after compute(const Matrix& A).
  /// \sa setDlsInverseThreshold(const Scalar), setMaxDamping(const Scalar)
  DlsInverseReturnType dlsInverse() {return DlsInverseReturnType(*this);}

  /// \brief Convenience alias for lsSolve(const Vector&)
  /// \sa lsSolve(const Vector&)
  LsSolveReturnType solve(const Vector& b) {return lsSolve(b);}

  /// TODO: Doc!
  /// \pre Must be called after compute(const Matrix& A).
  LsSolveReturnType lsSolve(const Vector& b) {return LsSolveReturnType(b, *this);}

  /// TODO: Doc!
  /// \pre Must be called after compute(const Matrix& A).
  DlsSolveReturnType dlsSolve(const Vector& b) {return DlsSolveReturnType(b, *this);}

  /// \param eps Threshold for smallest nonzero singular value used when computing the \e least-squares inverse.
  /// Valid values are in the range [0,1].
  void setLsInverseThreshold(const Scalar eps) {lsEps_ = eps;}

  /// \return Threshold for smallest nonzero singular value used when computing the \e least-squares inverse.
  /// \sa setLsInverseThreshold(const Scalar)
  Scalar getLsInverseThreshold() const {return lsEps_;}

  /**
   * \param dampEps Threshold for smallest nonzero singular value used when computing the \e damped \e least-squares
   * inverse. Valid values are in the range [0,1].
   * \note  The square of the damping factor \f$ \lambda^2 \f$ is computed as [1]:
   * \f[ \lambda^2 = \left\lbrace \begin{array}{ c l }
   * 0 & \mathrm{when} \;\; \sigma_m \geq eps \\
   * \left[ 1 - \left(\frac{\sigma_m}{dampEps}\right)^2 \right] \lambda_{max}^2 & \mathrm{otherwise}
   * \end{array} \right. \f]
   * where \f$ \sigma_m \f$ is the smallestsingular values of a matrix \f$ A \f$, and \f$ \lambda_{max} = dampMax \f$
   * is the maximum damping (see \ref setMaxDamping(const Scalar) ).
   *
   * [1] S. Chiaverini, O. Egeland, R.K. Kanestrom: Achieving user-deﬁned accuracy with damped least-squares inverse
   * kinematics, ICAR 1991 pp.  672–677
   * \sa setMaxDamping(const Scalar)
   */
  void setDlsInverseThreshold(const Scalar dampEps) {dlsEps_ = dampEps;}

  /// \return Threshold for smallest nonzero singular value used when computing the \e damped \e least-squares inverse.
  /// \sa setDlsInverseThreshold(const Scalar)
  Scalar getDlsInverseThreshold() const {return dlsEps_;}

  /// \param dampMax Maximum damping used when computing the \e damped \e least-squares inverse.
  /// \sa setDlsInverseThreshold(const Scalar)
  void setMaxDamping(const Scalar dampMax) {dampMax_ = dampMax;}

  /// \return Maximum damping used when computing the \e damped \e least-squares inverse.
  /// \sa setMaxDamping(const Scalar), setDlsInverseThreshold(const Scalar)
  Scalar getMaxDamping() const {return dampMax_;}

private:
  SVD       svd_;     ///< Singular Value Decomposition.
  Scalar    lsEps_;   ///< Threshold for smallest nonzero singular value used in least-squares inverse.
  Scalar    dlsEps_;  ///< Threshold for smallest nonzero singular value used in damped least-squares inverse.
  Scalar    dampMax_; ///< Maximum damping.

  // Preallocated instances to allow execution in contexts where heap allocations are forbidden
  Matrix    pinv_;   ///< Work matrix used when computing inverses explicitly.
  Vector    solve1_; ///< Work vector used when solving the associated linear system.
  Vector    solve2_; ///< Work vector used when solving the associated linear system.

  /// Initialize default parameters. This method is common to all constructors.
  void allocate(const Index rows, const Index cols);

  /// \return The number of singular values that are not approximately zero.
  Index nonzeroSingularValues() const;

  /// \param S Vector of singular values.
  /// \return Square of damping value.
  Scalar computeDampingSq(const Vector& S) const;

  friend class Eigen::internal::LsInverseReturnType<MatrixType>;
  friend class Eigen::internal::DlsInverseReturnType<MatrixType>;
  friend class Eigen::internal::LsSolveReturnType<MatrixType>;
  friend class Eigen::internal::DlsSolveReturnType<MatrixType>;
};

template <typename MatrixType>
MatrixInverter<MatrixType>::MatrixInverter(const Index rows, const Index cols)
{
  allocate(rows, cols);
}

template <typename MatrixType>
MatrixInverter<MatrixType>::MatrixInverter(const Matrix& A)
{
  allocate(A.rows(), A.cols());
  compute(A);
}

template <typename MatrixType>
inline void MatrixInverter<MatrixType>::allocate(const Index rows, const Index cols)
{
  if (rows == 0 || cols == 0) throw; // TODO: Throw what? Decide when to throw, when to assert

  svd_     = SVD(rows, cols, Eigen::ComputeThinU | Eigen::ComputeThinV);
  lsEps_ = Eigen::NumTraits<Scalar>::epsilon();
  dlsEps_ = 1e3 * Eigen::NumTraits<Scalar>::epsilon(); // TODO: Is this a sensible default?
  dampMax_ = 0.1;                                       // TODO: Is this a sensible default?
  const Index minDim  = std::min(rows, cols);
  pinv_   = Matrix(minDim, minDim);
  solve1_ = Vector(minDim);
  solve2_ = Vector(minDim);
}

template <typename MatrixType>
inline MatrixInverter<MatrixType>& MatrixInverter<MatrixType>::compute(const Matrix& A)
{
  // Check preconditions
  assert(A.rows() == svd_.rows() && A.cols() == svd_.cols() &&
         "Size mismatch between MatrixInverter and input matrix.");

  // Compute SVD of input matrix
  svd_.compute(A);
  return *this;
}

template <typename MatrixType>
inline typename MatrixInverter<MatrixType>::Index MatrixInverter<MatrixType>::nonzeroSingularValues() const
{
  const SVD&    svd  = svd_;
  const Vector& S    = svd.singularValues();
  const Index   end  = svd.nonzeroSingularValues(); // Number of singular values that are not _exactly_ zero

  Index id;
  // Loop going from smallest _nonzero_ singular value to first one larger than the prescribed tolerance
  for (id = end; id > 0; --id) { if (S(id -1) >= lsEps_ ) {break;} }
  return id;
}

template <typename MatrixType>
inline typename MatrixInverter<MatrixType>::Scalar
MatrixInverter<MatrixType>::computeDampingSq(const Vector& S) const
{
  const Scalar Smin        = S(S.size() - 1); // Smallest singular value
  return (Smin >= dlsEps_) ? 0.0 : (1.0 - (Smin * Smin) / (dlsEps_ * dlsEps_)) * (dampMax_ * dampMax_);
}

} // KDL


namespace Eigen
{
namespace internal
{

template<typename MatrixType>
struct traits<LsInverseReturnType<MatrixType> > {typedef typename MatrixType::PlainObject ReturnType;};

/// \brief Expression type for return value of MatrixInverter::lsInverse().
/// Compute the \e least-squares inverse of a matrix.
/// \tparam MatrixType Type of underlying dense matrix.
/// \sa class KDL::MatrixInverter
template<typename MatrixType>
class LsInverseReturnType : public Eigen::ReturnByValue<LsInverseReturnType<MatrixType> >
{
public:
  typedef MatrixType Matrix;
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;
  typedef typename KDL::MatrixInverter<MatrixType> MatrixInverter;
  typedef typename MatrixInverter::Vector Vector;
  typedef typename MatrixInverter::SVD SVD;

  /// \param inverter Inverter associated to this instance.
  LsInverseReturnType(MatrixInverter& inverter) : inverter_(inverter) {}

  /// Perform inverse computation.
  template <typename ResultType> void evalTo(ResultType& result) const;

  Index rows() const { return inverter_.svd_.cols(); } ///< \return Number of rows of the inverted matrix.
  Index cols() const { return inverter_.svd_.rows(); } ///< \return Number of columns of the inverted matrix.

private:
  MatrixInverter& inverter_; ///< Inverter associated to this instance.
};

template<typename MatrixType>
struct traits<DlsInverseReturnType<MatrixType> > {typedef typename MatrixType::PlainObject ReturnType;};

/// \brief Expression type for return value of MatrixInverter::dlsInverse().
/// Compute the \e damped \e least-squares inverse of a matrix.
/// \tparam MatrixType Type of underlying dense matrix.
/// \sa class KDL::MatrixInverter
template<typename MatrixType>
class DlsInverseReturnType : public Eigen::ReturnByValue<DlsInverseReturnType<MatrixType> >
{
public:
  typedef MatrixType Matrix;
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;
  typedef typename KDL::MatrixInverter<MatrixType> MatrixInverter;
  typedef typename MatrixInverter::Vector Vector;
  typedef typename MatrixInverter::SVD SVD;

  /// \param inverter Inverter associated to this instance.
  DlsInverseReturnType(MatrixInverter& inverter) : inverter_(inverter) {}

  /// Perform inverse computation.
  template <typename ResultType> void evalTo(ResultType& result) const;

  Index rows() const { return inverter_.svd_.cols(); } ///< \return Number of rows of the inverted matrix.
  Index cols() const { return inverter_.svd_.rows(); } ///< \return Number of columns of the inverted matrix.

private:
  MatrixInverter& inverter_; ///< Inverter associated to this instance.
};

template<typename MatrixType>
struct traits<LsSolveReturnType<MatrixType> > {typedef typename MatrixType::PlainObject ReturnType;};

/// \brief Expression type for return value of MatrixInverter::lsSolve().
/// Solve the linear system. TODO: Finish
/// \tparam MatrixType Type of underlying dense matrix.
/// \sa class KDL::MatrixInverter
template<typename MatrixType>
class LsSolveReturnType : public Eigen::ReturnByValue<LsSolveReturnType<MatrixType> >
{
public:
  typedef MatrixType Matrix;
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;
  typedef typename KDL::MatrixInverter<MatrixType> MatrixInverter;
  typedef typename MatrixInverter::Vector Vector;
  typedef typename MatrixInverter::SVD SVD;

  /// \param b Constant terms vector.
  /// \param inverter Inverter associated to this instance.
  LsSolveReturnType(const Vector&   b,
                    MatrixInverter& inverter) : b_(b), inverter_(inverter) {}

  /// Solve linear system.
  template <typename ResultType> void evalTo(ResultType& result) const;

  Index rows() const { return inverter_.svd_.cols(); } ///< \return Number of columns of the matrix to invert.
  Index cols() const { return 1; }

private:
  const Vector&   b_;        ///< Constant terms vector.
  MatrixInverter& inverter_; ///< Inverter associated to this instance.
};

template<typename MatrixType>
struct traits<DlsSolveReturnType<MatrixType> > {typedef typename MatrixType::PlainObject ReturnType;};

/// \brief Expression type for return value of MatrixInverter::dlsSolve().
/// Solve the linear system. TODO: Finish
/// \tparam MatrixType Type of underlying dense matrix.
/// \sa class KDL::MatrixInverter
template<typename MatrixType>
class DlsSolveReturnType : public Eigen::ReturnByValue<DlsSolveReturnType<MatrixType> >
{
public:
  typedef MatrixType Matrix;
  typedef typename Matrix::Index Index;
  typedef typename Matrix::Scalar Scalar;
  typedef typename KDL::MatrixInverter<MatrixType> MatrixInverter;
  typedef typename MatrixInverter::Vector Vector;
  typedef typename MatrixInverter::SVD SVD;

  /// \param b Constant terms vector.
  /// \param inverter Inverter associated to this instance.
  DlsSolveReturnType(const Vector&   b,
                     MatrixInverter& inverter) : b_(b), inverter_(inverter) {}

  /// Solve linear system.
  template <typename ResultType> void evalTo(ResultType& result) const;

  Index rows() const { return inverter_.svd_.cols(); } ///< \return Number of columns of the matrix to invert.
  Index cols() const { return 1; }

private:
  const Vector&   b_;        ///< Constant terms vector.
  MatrixInverter& inverter_; ///< Inverter associated to this instance.
};

template <typename MatrixType> template <typename ResultType>
void LsInverseReturnType<MatrixType>::evalTo(ResultType& result) const
{
  // Work matrix
  Matrix& T = inverter_.pinv_;

  // SVD details
  const SVD&  svd  = inverter_.svd_;
  const Index rows = svd.rows();
  const Index cols = svd.cols();

  const Vector& S = svd.singularValues(); // Sorted in decreasing order
  const Matrix& U = svd.matrixU();
  const Matrix& V = svd.matrixV();

  // Perform V Sinv U' subject to these optimizations:
  // - Space: Use the smallest possible temporaries. This is purpose of the (rows < cols) test
  // - Time:  Exclude unused blocks of U, S, V and work matrices from the computation. They correspond to the
  //   elements that are multiplied by the entries of S cutoff to zero.
  const Index cutoffSize = inverter_.nonzeroSingularValues();
  if (rows < cols)
  {
    T.topLeftCorner(cutoffSize, rows).noalias() =
    S.head(cutoffSize).array().inverse().matrix().asDiagonal() * U.leftCols(cutoffSize).transpose(); // Sinv U'
    result.noalias() = V.leftCols(cutoffSize) * T.topLeftCorner(cutoffSize, rows);                   // V Sinv U'
  }
  else
  {
    T.topLeftCorner(cols, cutoffSize).noalias() =
    V.leftCols(cutoffSize) * S.head(cutoffSize).array().inverse().matrix().asDiagonal();       // V Sinv
    result.noalias() = T.topLeftCorner(cols, cutoffSize) * U.leftCols(cutoffSize).transpose(); // V Sinv U'
  }
}

template <typename MatrixType> template <typename ResultType>
void DlsInverseReturnType<MatrixType>::evalTo(ResultType& result) const
{
  // Work matrix
  Matrix& T = inverter_.pinv_;

  // SVD details
  const SVD&  svd  = inverter_.svd_;
  const Index rows = svd.rows();
  const Index cols = svd.cols();

  const Vector& S = svd.singularValues(); // Sorted in decreasing order
  const Matrix& U = svd.matrixU();
  const Matrix& V = svd.matrixV();

  // Damped least-squares inverse: V Sinv U'
  const Scalar dampSq = inverter_.computeDampingSq(S);
  const Index  minDim = std::min(rows, cols);
  if (rows < cols)
  {
    T.topLeftCorner(minDim, minDim).noalias() =
    (S.array() / (S.array().square() + dampSq)).matrix().asDiagonal() * U.transpose(); // Sinv U'
    result.noalias() = V * T.topLeftCorner(minDim, minDim);                            // V Sinv U'
  }
  else
  {
    T.topLeftCorner(minDim, minDim).noalias() =
    V * (S.array() / (S.array().square() + dampSq)).matrix().asDiagonal();  // V Sinv
    result.noalias() = T.topLeftCorner(minDim, minDim) * U.transpose();     // V Sinv U'
  }
}

template <typename MatrixType> template <typename ResultType>
inline void LsSolveReturnType<MatrixType>::evalTo(ResultType& result) const
{
  // Work vectors
  Vector& vec1 = inverter_.solve1_;
  Vector& vec2 = inverter_.solve2_;

  // SVD details
  const SVD&    svd = inverter_.svd_;
  const Vector& S   = svd.singularValues(); // Sorted in decreasing order
  const Matrix& U   = svd.matrixU();
  const Matrix& V   = svd.matrixV();

  // Perform V Sinv U' b subject to these optimizations:
  // - Space, time: Favor matrix-vector products over matrix-matrix products.
  // - Time:  Exclude unused blocks of U, S, V and work matrices from the computation. They correspond to the
  //   elements that are multiplied by the entries of S cutoff to zero.
  const Index cutoffSize = inverter_.nonzeroSingularValues();

  vec1.head(cutoffSize).noalias() = U.leftCols(cutoffSize).transpose() * b_;                     // U' b
  vec2.head(cutoffSize).noalias() = S.head(cutoffSize).array().inverse().matrix().asDiagonal() *
                                    vec1.head(cutoffSize);                                       // Sinv U' b
  result.noalias()                = V.leftCols(cutoffSize) * vec2.head(cutoffSize);              // V Sinv U' b
}

template <typename MatrixType> template <typename ResultType>
inline void DlsSolveReturnType<MatrixType>::evalTo(ResultType& result) const
{
  // Work vectors
  Vector& vec1 = inverter_.solve1_;
  Vector& vec2 = inverter_.solve2_;

  // SVD details
  const SVD&    svd = inverter_.svd_;
  const Vector& S   = svd.singularValues(); // Sorted in decreasing order
  const Matrix& U   = svd.matrixU();
  const Matrix& V   = svd.matrixV();

  // Perform V Sinv U' b subject to these optimizations:
  // Space, time: Favor matrix-vector products over matrix-matrix products.
  const Scalar dampSq = inverter_.computeDampingSq(S);

  vec1.noalias()   = U.transpose() * b_;                                                       // U' b
  vec2.noalias()   = (S.array() / (S.array().square() + dampSq)).matrix().asDiagonal() * vec1; // Sinv U' b
  result.noalias() = V * vec2;                                                                 // V Sinv U' b
}

} // Eigen
} // internal

#endif // KDL_MATRIXINVERTER_H
