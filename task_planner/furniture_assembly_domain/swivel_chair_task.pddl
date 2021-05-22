(define (problem table-assembly-full)
	(:domain furniture-assembly)
; just fill in one possibility: connect column to base, then connect seat to column, to avoid multiple choice, which is hard to select from in current setting
	(:objects
		seat - chair-seat
		column - chair-leg
        base - chair-base
		seat-edge column-end base-leg - handle-part
		seat-connect column-screw - action-part
		column-connector base-hole - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor seat)
		(on-floor column)
		(on-floor base)

		(clear seat)
		(clear column)
		(clear base)

		(empty robot-gripper)

		(part-of seat-edge seat)
		(part-of column-end column)
		(part-of base-leg base)

		(part-of seat-connect seat)
        (part-of column-screw column)
        (part-of column-connector column)
        (part-of base-hole base)

		(affords-picking seat-edge)
		(affords-picking column-end)
		(affords-picking base-leg)

		(affords-inserting seat-connect)
		(affords-inserting-into seat column-connector column)
        
        (affords-inserting column-screw)
		(affords-inserting-into column base-hole base)
	)

	(:goal (and (inserted-into-base column base)
				(inserted-into-top seat column)
		   	)
	)
)