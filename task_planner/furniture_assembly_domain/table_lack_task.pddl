(define (problem table-assembly-full)
	(:domain furniture-assembly)

	(:objects
		leg1 leg2 leg3 leg4 - table-leg
		top - table-top
		leg1-base leg2-base leg3-base leg4-base - handle-part
		leg1-screw leg2-screw leg3-screw leg4-screw - action-part
		top-hole1 top-hole2 top-hole3 top-hole4 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor leg1)
		(on-floor leg2)
		(on-floor leg3)
		(on-floor leg4)
		(on-floor top)

		(clear leg1)
		(clear leg2)
		(clear leg3)
		(clear leg4)

		(empty robot-gripper)

		(part-of leg1-base leg1)
		(part-of leg1-screw leg1)
		(part-of leg2-base leg2)
		(part-of leg2-screw leg2)
		(part-of leg3-base leg3)
		(part-of leg3-screw leg3)
		(part-of leg4-base leg4)
		(part-of leg4-screw leg4)
		(part-of top-hole1 top)
		(part-of top-hole2 top)
		(part-of top-hole3 top)
		(part-of top-hole4 top)

		(affords-picking leg1-base)
		(affords-picking leg2-base)
		(affords-picking leg3-base)
		(affords-picking leg4-base)
		(affords-screwing leg1-screw)
		(affords-screwing leg2-screw)
		(affords-screwing leg3-screw)
		(affords-screwing leg4-screw)
		(affords-screwing-into leg1 top-hole1 top)
		(affords-screwing-into leg2 top-hole2 top)
		(affords-screwing-into leg3 top-hole3 top)
		(affords-screwing-into leg4 top-hole4 top)
	)

	(:goal (and (screwed-into leg1 top)
				(screwed-into leg2 top)
				(screwed-into leg3 top)
				(screwed-into leg4 top)
		   	)
	)
)