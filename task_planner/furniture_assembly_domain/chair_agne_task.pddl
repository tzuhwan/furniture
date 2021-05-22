(define (problem table-assembly-full)
	(:domain furniture-assembly)
	(:objects
		seat - chair-seat
		leg1 leg2 - chair-leg
		leg1-base leg2-base seat-base - handle-part
		leg1-connect leg2-connect - action-part
		seat-connector leg1-connector - tool-part
		robot-gripper1 robot-gripper2 - gripper
	)

	(:init
		(on-floor seat)
		(on-floor leg1)
		(on-floor leg2)

		(clear seat)
		(clear leg1)
		(clear leg2)

		(empty robot-gripper1)
		(empty robot-gripper2)

		(part-of seat-base seat)
		(part-of leg1-base leg1)
		(part-of leg2-base leg2)

		(part-of leg1-connect leg1)
        (part-of leg2-connect leg2)
        (part-of seat-connector seat)
        (part-of leg1-connector leg1)

		(affords-picking seat-base)
		(affords-picking leg1-base)
		(affords-picking leg2-base)

		(affords-inserting leg1-connect)
        (affords-inserting leg2-connect)
		(affords-inserting-into leg1 seat-connector seat)
        (affords-inserting-into leg2 leg1-connector leg1)
	)

	(:goal (and (connected-to leg1 seat)
				(connected-to leg2 leg1)
		   	)
	)
)