(define (problem table-assembly-full)
	(:domain furniture-assembly)

	(:objects
		leg1 leg2 leg3 leg4 - table-leg
		top - table-top
		leg1-base leg2-base leg3-base leg4-base - handle-part
		leg1-connect leg2-connect leg3-connect leg4-connect - action-part
		top-connector1 top-connector2 top-connector3 top-connector4 - tool-part
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
		(part-of leg1-connect leg1)
		(part-of leg2-base leg2)
		(part-of leg2-connect leg2)
		(part-of leg3-base leg3)
		(part-of leg3-connect leg3)
		(part-of leg4-base leg4)
		(part-of leg4-connect leg4)
		(part-of top-connector1 top)
		(part-of top-connector2 top)
		(part-of top-connector3 top)
		(part-of top-connector4 top)

		(affords-picking leg1-base)
		(affords-picking leg2-base)
		(affords-picking leg3-base)
		(affords-picking leg4-base)
		(affords-connecting leg1-connect)
		(affords-connecting leg2-connect)
		(affords-connecting leg3-connect)
		(affords-connecting leg4-connect)
		(affords-connecting-to leg1 top-connector1 top)
		(affords-connecting-to leg2 top-connector2 top)
		(affords-connecting-to leg3 top-connector3 top)
		(affords-connecting-to leg4 top-connector4 top)
	)

	(:goal (and (connected-to leg1 top)
				(connected-to leg2 top)
				(connected-to leg3 top)
				(connected-to leg4 top)
		   	)
	)
)