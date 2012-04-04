/*
 * Software License Agreement (Modified BSD License)
 *
 *  Copyright (c) 2012, PAL Robotics, S.L.
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

/** \author Adolfo Rodriguez Tsouroukdissian, Hilario Tome. */

#ifndef REEM_KINEMATICS_IK_SOLVER_
#define REEM_KINEMATICS_IK_SOLVER_

// C++ standard
#include <cassert>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

// Boost
#include <boost/scoped_ptr.hpp>

// Eigen
#include <Eigen/Dense>

// KDL
#include <kdl/jacobian.hpp>
#include <kdl/jntarray.hpp>

namespace KDL
{
  template <typename T> class MatrixInverter;
  class TreeFkSolverPos_recursive;
  class TreeJntToJacSolver;
}

namespace reem_kinematics_constraint_aware
{

// TODO: Make a template on floating point type?. Maybe YAGNI because KDL does not support this, as it's doubles everywhere.
class IkSolver
{
public:
  /// Kinematic coupling with reference frame
  enum EndpointCoupling
  {
    Position    = 1,
    Orientation = 2,
    Pose        = Position | Orientation
  };

  typedef std::vector<std::string>                     EndpointNameList;
  typedef std::vector<EndpointCoupling>                EndpointCouplingList;

  // TODO: Map endpoint names to tree segments!
  IkSolver(const KDL::Tree&            tree,
           const EndpointNameList&     endpoint_names,
           const EndpointCouplingList& endpoint_couplings);

  /// Conveniece constructor for the particular case of solving for the pose of a single endpoint.
  IkSolver(const KDL::Chain& chain);

  virtual ~IkSolver();

  // TODO: What's up with the max iterations?!
  bool solve(const KDL::JntArray&           q_current,
             const std::vector<KDL::Frame>& x_desired,
                   KDL::JntArray&           q_next);

  /// Conveniece overload for the particular case of solving for the pose of a single endpoint.
  /// \note This method will allocate heap memory for constructing a std::vector<KDL::Frame>.
  bool solve(const KDL::JntArray& q_current,
             const KDL::Frame&    x_desired,
                   KDL::JntArray& q_next);

  // TODO: Add getters
  void setJointPositionLimits(const Eigen::VectorXd& q_min, const Eigen::VectorXd& q_max);
  void setMaxDeltaTwist(double delta_twist_max) {delta_twist_max_ = delta_twist_max;}
  void setMaxDeltaJointPos(double delta_joint_pos_max) {delta_joint_pos_max_ = delta_joint_pos_max;}
  void setVelocityIkGain(double velik_gain) {velik_gain_ = velik_gain;}
  void setEpsilon(double eps) {eps_ = eps;}
  void setMaxIterations(std::size_t max_iter) {max_iter_ = max_iter;}
  void setPosture(const KDL::JntArray& q_posture) {q_posture_ = q_posture;} ///< \note Copies values.
  void setJointSpaceWeights(const Eigen::VectorXd& Wq) {Wqinv_ref_ = Wq; Wqinv_ = Wq;} ///< Contains diagonal elements of matrix (nondiagonal not yet supported)

private:
  typedef std::vector<std::size_t>             CoupledDirections; /// 0,1,2 -> xyz translation, 3,4,5 -> xyz rotation
  typedef KDL::TreeFkSolverPos_recursive       FkSolver;
  typedef boost::scoped_ptr<FkSolver>          FkSolverPtr;
  typedef KDL::TreeJntToJacSolver              JacSolver;
  typedef boost::scoped_ptr<JacSolver>         JacSolverPtr;
  typedef KDL::MatrixInverter<Eigen::MatrixXd> Inverter;
  typedef boost::scoped_ptr<Inverter>          InverterPtr;
  typedef std::map<std::string, int>           JointNameToIndexMap;

  FkSolverPtr  fk_solver_;  ///< Forward kinematics solver
  JacSolverPtr jac_solver_; ///< Jacobian solver
  InverterPtr  inverter_;   ///< SVD matrix inverter/solver

  std::vector<std::string>       endpoint_names_;
  std::vector<CoupledDirections> coupled_dirs_;
  JointNameToIndexMap            joint_name_to_idx_; ///< Map from joint names to KDL tree indices // TODO: Remove???

  Eigen::VectorXd delta_twist_;
  Eigen::VectorXd delta_q_;
  Eigen::VectorXd q_min_; ///< Minimum joint position limits.
  Eigen::VectorXd q_max_; ///< Maximum joint position limits.
  Eigen::VectorXd q_tmp_;

  Eigen::MatrixXd jacobian_;
  KDL::Jacobian   jacobian_tmp_;

  // Custom velocity-IK members NOTE: Remove this and substitute with generic implementation
  Eigen::VectorXd Wqinv_ref_; ///< Inverse of joint space weight matrix (vector containing diagonal elements), static reference values
  Eigen::VectorXd Wqinv_;     ///< Equal to Wqinv_ref_, but with zeroed-out elements if joint limits are reached
  Eigen::MatrixXd nullspace_projector_;
  Eigen::MatrixXd identity_qdim_;
  KDL::JntArray   q_posture_;

  // Position IK configuration parameters
  double      delta_twist_max_;
  double      delta_joint_pos_max_;
  double      velik_gain_;
  double      eps_;
  std::size_t max_iter_;

  void init(const KDL::Tree&                     tree,
            const std::vector<std::string>&      endpoint_names,
            const std::vector<EndpointCoupling>& endpoint_couplings);

  void updateDeltaTwist(const KDL::JntArray& q, const std::vector<KDL::Frame>& x_desired);

  void updateJacobian(const KDL::JntArray& q);
};

inline void IkSolver::setJointPositionLimits(const Eigen::VectorXd& q_min, const Eigen::VectorXd& q_max)
{
  assert(q_min.size() == q_min_.size());
  assert(q_max.size() == q_max_.size());

  q_min_ = q_min;
  q_max_ = q_max;
}

inline bool IkSolver::solve(const KDL::JntArray& q_current,
                            const KDL::Frame&    x_desired,
                                  KDL::JntArray& q_next)
{
  return solve(q_current,
               std::vector<KDL::Frame>(1, x_desired),
               q_next);
}

} // reem_kinematics_constraint_aware

#endif // REEM_KINEMATICS_IK_SOLVER_