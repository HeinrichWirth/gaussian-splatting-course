import {mountScene,mountBell} from './labs/foundations.js';
import {mountPlane,mountCovariance} from './labs/plane.js';
import {mountEditor} from './labs/editor.js';
import {mountLoss,mountDescent,mountAutograd,mountFitting} from './labs/learning.js';
import {mountFrames, mountPerspective, mountEllipsoid, mountFootprint} from './labs/geometry3d.js';
import {mountVisibility, mountHarmonics, mountMultiview, mountDensity} from './labs/render-training.js';
import {mountTiles, mountCapture, mountBuildScene, mountEvaluation} from './labs/reconstruction.js';
import {mountMotionViews, mountTrajectory, mountPersistent} from './labs/motion.js';
import {mountDeformField, mountDynamicFit} from './labs/deformation.js';
import {mountStorageBudget, mountStaticCodec, mountMotionCodec, mountCodecStudy} from './labs/compression.js';

// The disclosure beside each experiment opens the same module that draws it.
export const experiments = {
  scene: [mountScene, 'foundations'], bell: [mountBell, 'foundations'],
  plane: [mountPlane, 'plane'], covariance: [mountCovariance, 'plane'], editor: [mountEditor, 'editor'],
  loss: [mountLoss, 'learning'], descent: [mountDescent, 'learning'],
  autograd: [mountAutograd, 'learning'], fitting: [mountFitting, 'learning'],
  frames: [mountFrames, 'geometry3d'], perspective: [mountPerspective, 'geometry3d'],
  ellipsoid: [mountEllipsoid, 'geometry3d'], footprint: [mountFootprint, 'geometry3d'],
  visibility: [mountVisibility, 'render-training'], harmonics: [mountHarmonics, 'render-training'],
  multiview: [mountMultiview, 'render-training'], density: [mountDensity, 'render-training'],
  tiles: [mountTiles, 'reconstruction'], capture: [mountCapture, 'reconstruction'],
  buildscene: [mountBuildScene, 'reconstruction'], evaluation: [mountEvaluation, 'reconstruction'],
  'motion-views': [mountMotionViews, 'motion'], trajectory: [mountTrajectory, 'motion'],
  persistent: [mountPersistent, 'motion'], 'deform-field': [mountDeformField, 'deformation'],
  'dynamic-fit': [mountDynamicFit, 'deformation'], storageBudget: [mountStorageBudget, 'compression'],
  staticCodec: [mountStaticCodec, 'compression'], motionCodec: [mountMotionCodec, 'compression'],
  codecStudy: [mountCodecStudy, 'compression'],
};
