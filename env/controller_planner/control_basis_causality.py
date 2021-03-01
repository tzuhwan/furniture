"""
Defines causality of controllers in control basis.
"""

class ControlBasis:

	def __init__(self):
		self.controllers = [
			"Baxter6DPoseController",
			"Baxter3DPositionController",
			"BaxterAlignmentController",
			"BaxterRotationController",
			"BaxterObject6DPoseController"
		]

		self.causal_models = {
			"Baxter6DPoseController": 0.95,
			"Baxter3DPositionController": 0.90,
			"BaxterAlignmentController": 0.85,
			"BaxterRotationController": 0.97,
			"BaxterObject6DPoseController": 0.80
		}

		self.temporal_decomposition = {
			"screw-into": ["Baxter3DPositionController", "BaxterAlignmentController", "BaxterRotationController"],
			"insert-into": ["Baxter3DPositionController", "BaxterAlignmentController"]
		}

	def get_controllers(self):
		return self.controllers

	def get_causal_models(self):
		return self.causal_models

	def get_temporal_decomposition(self):
		return self.temporal_decomposition
		