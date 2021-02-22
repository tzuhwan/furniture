(define (problem table-assembly-full)
	(:domain furniture-assembly)

	(:objects
		table-leg1 table-leg2 table-leg3 table-leg4 - table-leg
		table-lack-top - table-top
		table-leg1-base table-leg2-base table-leg3-base table-leg4-base - handle-part
		table-leg1-screw table-leg2-screw table-leg3-screw table-leg4-screw - action-part
		table-lack-top-hole1 table-lack-top-hole2 table-lack-top-hole3 table-lack-top-hole4 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor table-leg1)
		(on-floor table-leg2)
		(on-floor table-leg3)
		(on-floor table-leg4)
		(on-floor table-lack-top)

		(clear table-leg1)
		(clear table-leg2)
		(clear table-leg3)
		(clear table-leg4)

		(empty robot-gripper)

		(part-of table-leg1-base table-leg1)
		(part-of table-leg1-screw table-leg1)
		(part-of table-leg2-base table-leg2)
		(part-of table-leg2-screw table-leg2)
		(part-of table-leg3-base table-leg3)
		(part-of table-leg3-screw table-leg3)
		(part-of table-leg4-base table-leg4)
		(part-of table-leg4-screw table-leg4)
		(part-of table-lack-top-hole1 table-lack-top)
		(part-of table-lack-top-hole2 table-lack-top)
		(part-of table-lack-top-hole3 table-lack-top)
		(part-of table-lack-top-hole4 table-lack-top)

		(affords-picking table-leg1-base)
		(affords-picking table-leg2-base)
		(affords-picking table-leg3-base)
		(affords-picking table-leg4-base)
		(affords-screwing table-leg1-screw)
		(affords-screwing table-leg2-screw)
		(affords-screwing table-leg3-screw)
		(affords-screwing table-leg4-screw)
		(affords-screwing-into table-leg1 table-lack-top-hole1 table-lack-top)
		(affords-screwing-into table-leg2 table-lack-top-hole2 table-lack-top)
		(affords-screwing-into table-leg3 table-lack-top-hole3 table-lack-top)
		(affords-screwing-into table-leg4 table-lack-top-hole4 table-lack-top)
	)

	(:goal (and (screwed-into table-leg1 table-lack-top)
				(screwed-into table-leg2 table-lack-top)
				(screwed-into table-leg3 table-lack-top)
				(screwed-into table-leg4 table-lack-top)
		   	)
	)
)