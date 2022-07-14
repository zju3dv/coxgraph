/*
 * Copyright (c) 2018, Vision for Robotics Lab
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * * Redistributions of source code must retain the above copyright
 * notice, this list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 * * Neither the name of the Vision for Robotics Lab, ETH Zurich nor the
 * names of its contributors may be used to endorse or promote products
 * derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 */

/*
 * keyframe-database.cpp
 * @brief Source for the KeyFrameDatabase class.
 * @author: Marco Karrer
 * Created on: Aug 15, 2018
 */

#include "pose_graph_backend/keyframe-database.hpp"

#include <coxgraph_mod/vio_interface.h>

namespace pgbe {

KeyFrameDatabase::KeyFrameDatabase(const SystemParameters& params)
    : parameters_(params) {
  inverted_file_.resize(parameters_.voc_ptr->size());
}

void KeyFrameDatabase::add(std::shared_ptr<KeyFrame> keyframe_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);

  for (DBoW2::BowVector::const_iterator it = keyframe_ptr->bow_vec_.begin();
       it != keyframe_ptr->bow_vec_.end(); ++it) {
    inverted_file_[it->first].push_back(keyframe_ptr);
  }
}

void KeyFrameDatabase::erase(std::shared_ptr<KeyFrame> keyframe_ptr) {
  std::unique_lock<std::mutex> lock(mutex_);

  for (DBoW2::BowVector::const_iterator it = keyframe_ptr->bow_vec_.begin();
       it != keyframe_ptr->bow_vec_.end(); ++it) {
    // List of keyframes that share the word
    KFlist& list_of_kfs = inverted_file_[it->first];

    for (KFlist::iterator lit = list_of_kfs.begin(); lit != list_of_kfs.end();
         ++it) {
      if (keyframe_ptr == (*lit)) {
        list_of_kfs.erase(lit);
        break;
      }
    }
  }
}

void KeyFrameDatabase::clear() {
  inverted_file_.clear();
  inverted_file_.resize(parameters_.voc_ptr->size());
}

std::vector<std::shared_ptr<KeyFrame>,
            Eigen::aligned_allocator<std::shared_ptr<KeyFrame>>>
KeyFrameDatabase::detectLoopCandidates(std::shared_ptr<KeyFrame> query,
                                       KFset connected_kfs,
                                       // TODO: KFset all_kfs_in_map,
                                       const double min_score,
                                       const int max_loop_candidates) {
  KFlist kfs_sharing_words;
  std::map<std::shared_ptr<KeyFrame>, int> common_word_count;

  {
    std::unique_lock<std::mutex> lock(mutex_);
    for (DBoW2::BowVector::const_iterator it = query->bow_vec_.begin();
         it != query->bow_vec_.end(); ++it) {
      KFlist& listed_kfs = inverted_file_[it->first];
      for (KFlist::iterator lit = listed_kfs.begin(); lit != listed_kfs.end();
           ++lit) {
        std::shared_ptr<KeyFrame> kf_ptr = (*lit);
        if (kf_ptr->getId() == query->getId()) continue;

        // if kf from same client, continue
        if (kf_ptr->getId().first == query->getId().first) continue;

        // if not need to fuse, continue
        // if (!coxgraph::mod::needToFuse(kf_ptr->getId().first,
        //                                query->getId().first))
        //   continue;

        // TODO: if (!all_kfs_in_map.count(kf_ptr)) continue;
        auto itr = std::find(kfs_sharing_words.begin(), kfs_sharing_words.end(),
                             kf_ptr);
        if (itr != kfs_sharing_words.end()) continue;

        if (!connected_kfs.count(kf_ptr)) {
          kfs_sharing_words.push_back(kf_ptr);
        }

        if (common_word_count.count(kf_ptr)) {
          ++common_word_count[kf_ptr];
        } else {
          common_word_count.insert(std::make_pair(kf_ptr, 1));
        }
      }
    }
  }

  if (kfs_sharing_words.empty()) {
    return KFvec();
  }

  int max_common_words = 0;
  for (KFlist::iterator it = kfs_sharing_words.begin();
       it != kfs_sharing_words.end(); ++it) {
    std::shared_ptr<KeyFrame> tmp_kf = (*it);
    if (!common_word_count.count(tmp_kf)) continue;

    if (common_word_count[tmp_kf] > max_common_words) {
      max_common_words = common_word_count[tmp_kf];
    }
  }

  int min_common_words = 0.5f * max_common_words;

  std::list<std::pair<float, std::shared_ptr<KeyFrame>>> score_and_match;
  int nscore = 0;

  for (KFlist::iterator it = kfs_sharing_words.begin();
       it != kfs_sharing_words.end(); ++it) {
    std::shared_ptr<KeyFrame> tmp_kf = (*it);

    if (!common_word_count.count(tmp_kf)) continue;

    if (common_word_count[tmp_kf] > min_common_words) {
      ++nscore;

      float si = parameters_.voc_ptr->score(query->bow_vec_, tmp_kf->bow_vec_);

      if (std::abs(query->getTimestamp() - tmp_kf->getTimestamp()) < 5.0) {
        continue;
      }

      if (si >= min_score) {
        score_and_match.push_back(std::make_pair(si, tmp_kf));
      }
    }
  }

  if (score_and_match.empty()) {
    return KFvec();
  }

  // only return a max number of loop candidates with the best score
  // first sort by descending score
  score_and_match.sort([](std::pair<float, std::shared_ptr<KeyFrame>>& a,
                          std::pair<float, std::shared_ptr<KeyFrame>>& b) {
    return a.first > b.first;
  });

  std::list<std::pair<float, std::shared_ptr<KeyFrame>>>::iterator
      max_loop_candidates_iter;
  if (score_and_match.size() > max_loop_candidates) {
    max_loop_candidates_iter =
        std::next(score_and_match.begin(), max_loop_candidates);
  } else {
    max_loop_candidates_iter = score_and_match.end();
  }

  KFvec loop_candidates;
  loop_candidates.reserve(score_and_match.size());
  for (auto it = score_and_match.begin(); it != max_loop_candidates_iter;
       ++it) {
    auto itr =
        std::find(loop_candidates.begin(), loop_candidates.end(), it->second);
    if (itr == loop_candidates.end()) {
      loop_candidates.push_back((it->second));
    }
  }

  return loop_candidates;
}

}  // namespace pgbe
