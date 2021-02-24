(define (problem table-assembly-full)
	(:domain furniture-assembly)

	(:objects
		table-leg1 table-leg2 table-leg3 table-leg4 - table-leg
		table-klubbo-top - table-top
		table-leg1-base table-leg2-base table-leg3-base table-leg4-base - handle-part
		table-leg1-connect table-leg2-connect table-leg3-connect table-leg4-connect - action-part
		table-klubbo-top-connector1 table-klubbo-top-connector2 table-klubbo-top-connector3 table-klubbo-top-connector4 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor table-leg1)
		(on-floor table-leg2)
		(on-floor table-leg3)
		(on-floor table-leg4)
		(on-floor table-klubbo-top)

		(clear table-leg1)
		(clear table-leg2)
		(clear table-leg3)
		(clear table-leg4)

		(empty robot-gripper)

		(part-of table-leg1-base table-leg1)
		(part-of table-leg1-connect table-leg1)
		(part-of table-leg2-base table-leg2)
		(part-of table-leg2-connect table-leg2)
		(part-of table-leg3-base table-leg3)
		(part-of table-leg3-connect table-leg3)
		(part-of table-leg4-base table-leg4)
		(part-of table-leg4-connect table-leg4)
		(part-of table-klubbo-top-connector1 table-klubbo-top)
		(part-of table-klubbo-top-connector2 table-klubbo-top)
		(part-of table-klubbo-top-connector3 table-klubbo-top)
		(part-of table-klubbo-top-connector4 table-klubbo-top)

		(affords-picking table-leg1-base)
		(affords-picking table-leg2-base)
		(affords-picking table-leg3-base)
		(affords-picking table-leg4-base)
		(affords-connecting table-leg1-connect)
		(affords-connecting table-leg2-connect)
		(affords-connecting table-leg3-connect)
		(affords-connecting table-leg4-connect)
		(affords-connecting-to table-leg1 table-klubbo-top-connector1 table-klubbo-top)
		(affords-connecting-to table-leg2 table-klubbo-top-connector2 table-klubbo-top)
		(affords-connecting-to table-leg3 table-klubbo-top-connector3 table-klubbo-top)
		(affords-connecting-to table-leg4 table-klubbo-top-connector4 table-klubbo-top)
	)

	(:goal (and (connected-to table-leg1 table-klubbo-top)
				(connected-to table-leg2 table-klubbo-top)
				(connected-to table-leg3 table-klubbo-top)
				(connected-to table-leg4 table-klubbo-top)
		   	)
	)
)