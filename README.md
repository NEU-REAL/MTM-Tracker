# Motion-to-Matching: A Mixed Paradigm for 3D Single Object Tracking (RAL 2023)

This is the official code release of "Motion-to-Matching: A Mixed Paradigm for 3D Single Object Tracking"

<embed src="https://github.com/LeoZhiheng/PaperReading/blob/main/Picture/MTM-Tracker.pdf" type="application/pdf" width="100%" height="600px" />

## Abstract

3D single object tracking with LiDAR points is an important task in the computer vision field. Previous methods usually adopt the matching-based or motion-centric paradigms to estimate the current target status. However, the former is sensitive to the similar distractors and the sparseness of point clouds due to relying on appearance matching, while the latter usually focuses on short-term motion clues (eg. two frames) and ignores the long-term motion pattern of target. To address these issues, we propose a mixed paradigm with two stages, named **MTM-Tracker**, which combines motion modeling with feature matching into a single network. Specifically, in the first stage, we exploit the continuous historical boxes as motion prior and propose an encoder-decoder structure to locate target coarsely. Then, in the second stage, we introduce a feature interaction module to extract motion-aware features from consecutive point clouds and match them to refine target movement as well as regress other target states. Extensive experiments validate that our paradigm achieves competitive performance on large-scale datasets (70.9\% in KITTI and 51.70\% in NuScenes).

## Method

## Code
Coming soon...
