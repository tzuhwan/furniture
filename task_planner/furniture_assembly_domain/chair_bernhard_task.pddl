(define (problem table-assembly-full)
	(:domain furniture-assembly)
	(:objects
		seat - chair-seat
		left-leg right-leg - chair-leg
		left-leg-base right-leg-base seat-base - handle-part
		left-leg-connect right-leg-connect - action-part
		seat-connector-left seat-connector-right - tool-part
		robot-gripper1 robot-gripper2 - gripper
	)

	(:init
		(on-floor seat)
		(on-floor left-leg)
		(on-floor right-leg)

		(clear seat)
		(clear left-leg)
		(clear right-leg)

		(empty robot-gripper1)
		(empty robot-gripper2)

		(part-of seat-base seat)
		(part-of left-leg-base left-leg)
		(part-of right-leg-base right-leg)

		(part-of left-leg-connect left-leg)
        (part-of right-leg-connect right-leg)
        (part-of seat-connector-left seat)
        (part-of seat-connector-right seat)

		(affords-picking seat-base)
		(affords-picking left-leg-base)
		(affords-picking right-leg-base)

		(affords-inserting left-leg-connect)
        (affords-inserting right-leg-connect)
		(affords-inserting-into left-leg seat-connector-left seat)
        (affords-inserting-into right-leg seat-connector-right seat)
	)

	(:goal (and (connected-to left-leg seat)
				(connected-to right-leg seat)
		   	)
	)
)