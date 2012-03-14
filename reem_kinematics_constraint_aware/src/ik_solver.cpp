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

// KDL
#include <kdl/chain.hpp>
#include <kdl/tree.hpp>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>

// Local
#include <reem_kinematics_constraint_aware/matrix_inverter.h>
#include <reem_kinematics_constraint_aware/ik_solver.h>

#include <ros/ros.h> // TODO: Remove!
using namespace reem_kinematics_constraint_aware;
using std::size_t;
using Eigen::VectorXd;

IkSolver::IkSolver(const KDL::Tree&            tree,
                   const EndpointNameList&     endpoint_names,
                   const EndpointCouplingList& endpoint_couplings)
{
  init(tree, endpoint_names, endpoint_couplings);
}

IkSolver::IkSolver(const KDL::Chain& chain)
{
  const std::string root_name = chain.getSegment(0).getName();
  const std::string tip_name  = chain.getSegment(chain.getNrOfSegments() - 1).getName();

  KDL::Tree tree("root");
  tree.addChain(chain, "root");

  std::vector<std::string> endpoint_names(1, tip_name);
  std::vector<EndpointCoupling> endpoint_couplings(1, Pose);

  init(tree, endpoint_names, endpoint_couplings);
}

IkSolver::~IkSolver() {}

void IkSolver::init(const KDL::Tree&                     tree,
                    const std::vector<std::string>&      endpoint_names,
                    const std::vector<EndpointCoupling>& endpoint_couplings)
{
  // Preconditions
  assert(endpoint_names.size() == endpoint_couplings.size() && "endpoints and coupling vectors size mismatch");
  // TODO: Verify that all endpoints are contained in tree

  // Enpoint names vector
  endpoint_names_ = endpoint_names;

  // Problem size
  const size_t q_dim = tree.getNrOfJoints(); // Joint space dimension
  size_t x_dim = 0;                          // Task space dimension, value assigned below

  // Populate coupled directions vector
  coupled_dirs_.resize(endpoint_names_.size());
  for (size_t i = 0; i < coupled_dirs_.size(); ++i)
  {
    if ((endpoint_couplings[i] & Position) == Position)
    {
      coupled_dirs_[i].push_back(0);
      coupled_dirs_[i].push_back(1);
      coupled_dirs_[i].push_back(2);
      x_dim += 3;
    }
    if ((endpoint_couplings[i] & Orientation) == Orientation)
    {
      coupled_dirs_[i].push_back(3);
      coupled_dirs_[i].push_back(4);
      coupled_dirs_[i].push_back(5);
      x_dim += 3;
    }
  }

  // Initialize kinematics solvers
  fk_solver_.reset(new FkSolver(tree));
  jac_solver_.reset(new JacSolver(tree));
  inverter_.reset(new Inverter(x_dim, q_dim));

  // Matrix inversion parameters TODO: Expose!
  inverter_->setLsInverseThreshold(1e-5);
  inverter_->setDlsInverseThreshold(1e-4);
  inverter_->setMaxDamping(0.05);

  // Default values of position solver parameters
  delta_twist_max_ = Eigen::NumTraits<double>::highest();
  velik_gain_      = 1.0;
  eps_             = Eigen::NumTraits<double>::epsilon();
  max_iter_        = 1;

  // Populate map from joint names to KDL tree indices
  joint_name_to_idx_.clear();
  const KDL::SegmentMap& tree_segments = tree.getSegments();
  for (KDL::SegmentMap::const_iterator it = tree_segments.begin(); it != tree_segments.end(); ++it)
  {
    const KDL::Joint& joint = it->second.segment.getJoint();
    if (joint.getType() != KDL::Joint::None)
    {
      joint_name_to_idx_[joint.getName()] = it->second.q_nr;
    }
  }

  // Preallocate IK resources
  delta_twist_ = VectorXd::Zero(x_dim);
  delta_q_     = VectorXd::Zero(q_dim);
  q_min_       = VectorXd::Constant(q_dim, Eigen::NumTraits<double>::lowest());  // If joint limits are not set, any
  q_max_       = VectorXd::Constant(q_dim, Eigen::NumTraits<double>::highest()); // representable joint value is valid

  jacobian_     = Eigen::MatrixXd(x_dim, q_dim);
  jacobian_tmp_ = KDL::Jacobian(q_dim);

  q_posture_           = Eigen::VectorXd::Zero(q_dim);
  nullspace_projector_ = Eigen::MatrixXd(q_dim, q_dim);
  identity_qdim_       = Eigen::MatrixXd::Identity(q_dim, q_dim);
  Wqinv_               = Eigen::VectorXd::Ones(q_dim);
  Wqinv_(joint_name_to_idx_["torso_1_joint"]) = 0.1;  // TODO: Read from config file
  Wqinv_(joint_name_to_idx_["torso_2_joint"]) = 0.1;
  q_posture_ = Eigen::VectorXd::Zero(q_dim);
  q_posture_(joint_name_to_idx_["arm_right_1_joint"]) = -0.4;
  q_posture_(joint_name_to_idx_["arm_right_2_joint"]) =  0.6;
  q_posture_(joint_name_to_idx_["arm_right_3_joint"]) = -0.1;
  q_posture_(joint_name_to_idx_["arm_right_4_joint"]) =  0.6109;
  q_posture_(joint_name_to_idx_["arm_right_5_joint"]) =  0.2;
  q_posture_(joint_name_to_idx_["arm_right_6_joint"]) =  0.4;
  q_posture_(joint_name_to_idx_["arm_right_7_joint"]) =  0.2;
}

bool IkSolver::solve(const KDL::JntArray&           q_current,
                     const std::vector<KDL::Frame>& x_desired,
                           KDL::JntArray&           q_next)
{
  // Precondition
  assert(endpoint_names_.size() == x_desired.size());

  q_next = q_current;
  size_t i;
  for (i = 0; i < max_iter_; ++i)
  {
    // Update current task space velocity error
    updateDeltaTwist(q_next, x_desired);
    if (delta_twist_.norm() < eps_) {break;}

    // Update Jacobian
    updateJacobian(q_next);

    // Velocity IK: Compute incremental joint displacement and scale according to gain

    // Prepare computation of IK with nullspace optimization
    using Eigen::MatrixXd;
    using Eigen::VectorXd;
    using Eigen::DiagonalWrapper;

    const MatrixXd& J = jacobian_;                                 // Convenience alias
    const DiagonalWrapper<const VectorXd> W = Wqinv_.asDiagonal(); // Convenience alias

    // Perform SVD decomposition of J W
    inverter_->compute(J * W);

    // Nullspace projector
    nullspace_projector_ = identity_qdim_ - W * inverter_->inverse() * J; // NOTE: Not rt-friendly, allocates temporaries

    // Compute incremental joint displacement
    delta_q_  = W * inverter_->dlsSolve(delta_twist_) + nullspace_projector_ * (q_posture_ - q_next.data);
    delta_q_ *= velik_gain_;

    // Integrate joint velocity
    q_next.data += delta_q_;

    // Enforce joint position limits
    typedef Eigen::VectorXd::Index Index;
    for(Index j = 0; j < q_min_.size(); ++j)
    {
      if(q_next(j) < q_min_(j)) {q_next(j) = q_min_(j);}
    }

    for(Index j = 0; j < q_max_.size(); ++j)
    {
      if(q_next(j) > q_max_(j)) {q_next(j) = q_max_(j);}
    }

  }
//   ROS_WARN_STREAM("i = " << i << " <= " << max_iter_ << "\n" << q_next(0) << " " << q_next(1) << " " <<q_next(2) << " " <<q_next(3) << " " <<q_next(4) << " " << q_next(5)<< " " << q_next(6)<< " " << q_next(7)<< " " << q_next(8));
//   ROS_WARN_STREAM("delta_twist " << delta_twist_.transpose());
//   ROS_ERROR_STREAM("return " << (i < max_iter_));
  return (i < max_iter_);
}

void IkSolver::updateDeltaTwist(const KDL::JntArray& q, const std::vector<KDL::Frame>& x_desired)
{
  KDL::Frame ith_frame;
  KDL::Twist ith_delta_twist;
  size_t x_idx = 0;
  for (size_t i = 0; i < endpoint_names_.size(); ++i)
  {
    // Forward kinematics of ith endpoint
    fk_solver_->JntToCart(q, ith_frame, endpoint_names_[i]);
    ith_delta_twist = diff(ith_frame, x_desired[i]);

    KDL::Vector rot   = ith_frame.M.GetRot();
    KDL::Vector rot_d = x_desired[i].M.GetRot();
//     ROS_ERROR_STREAM("q " << q.data);
//     ROS_ERROR_STREAM("Curr " << ith_frame.p[0] << " " << ith_frame.p[1] << " " << ith_frame.p[2] << ", " << rot[0] << " " << rot[1] << " " << rot[2]);
//     ROS_ERROR_STREAM("Desi " << x_desired[i].p[0] << " " << x_desired[i].p[1] << " " << x_desired[i].p[2] << ", " << rot_d[0] << " " << rot_d[1] << " " << rot_d[2]);
    // Extract only task-space directions relevant to the IK problem
    const CoupledDirections& endpoint_coupled_dirs = coupled_dirs_[i];
    for (size_t j = 0; j < endpoint_coupled_dirs.size(); ++j)
    {
      delta_twist_(x_idx) = ith_delta_twist[endpoint_coupled_dirs[j]];
      ++x_idx;
    }
  }
//   ROS_ERROR_STREAM("delta_twist " << delta_twist_.transpose());

  // Enforce task space maximum velocity through uniform scaling
  double delta_twist_scaling = 1.0;
  typedef Eigen::VectorXd::Index Index;
  for (Index i = 0; i < delta_twist_.size(); ++i)
  {
    const double current_scaling = delta_twist_max_ / std::abs(delta_twist_(i));
    if (current_scaling < delta_twist_scaling)
    {
      delta_twist_scaling = current_scaling;
    }
  }
  if (delta_twist_scaling < 1.0)
  {
    delta_twist_ *= delta_twist_scaling;
  }
//   ROS_ERROR_STREAM("delta_twist " << delta_twist_.transpose());
//   ROS_ERROR("-----");
}

void IkSolver::updateJacobian(const KDL::JntArray& q)
{
  size_t x_idx = 0;

  for (size_t i = 0; i < endpoint_names_.size(); ++i)
  {
    // Jacobian of ith endpoint
    jac_solver_->JntToJac(q, jacobian_tmp_, endpoint_names_[i]);

    // Extract only task-space directions (Jacobian rows) relevant to the IK problem
    const CoupledDirections& endpoint_coupled_dirs = coupled_dirs_[i];
    for (size_t j = 0; j < endpoint_coupled_dirs.size(); ++j)
    {
      jacobian_.row(x_idx) = jacobian_tmp_.data.row(endpoint_coupled_dirs[j]);
      ++x_idx;
    }
  }
}