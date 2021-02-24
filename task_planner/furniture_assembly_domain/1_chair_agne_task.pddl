(define (problem table-assembly-full)
	(:domain furniture-assembly)
; todo: add a conditional connection (leg2 and seat can connect only if leg1 is already connected to seat. don't know how to express this in domain.pddl)
	(:objects
		seat - chair-seat
		leg1 leg2 - chair-leg
		leg1-base leg2-base seat-base - handle-part
		leg1-connect leg2-connect - action-part
		seat-connector1 seat-connector2 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor seat)
		(on-floor leg1)
		(on-floor leg2)

		(clear seat)
		(clear leg1)
		(clear leg2)

		(empty robot-gripper)

		(part-of seat-base seat)
		(part-of leg1-base leg1)
		(part-of leg2-base leg2)

		(part-of leg1-connect leg1)
        (part-of leg2-connect leg2)
        (part-of seat-connector1 seat)
        (part-of seat-connector2 seat)

		(affords-picking seat-base)
		(affords-picking leg1-base)
		(affords-picking leg2-base)

		(affords-connecting leg1-connect)
        (affords-connecting leg2-connect)
		(affords-connecting-to leg1 seat-connector1 seat)
        (affords-connecting-to leg2 seat-connector2 seat)
	)

	(:goal (and (connected-to leg1 seat)
				(connected-to leg2 seat)
		   	)
	)
)