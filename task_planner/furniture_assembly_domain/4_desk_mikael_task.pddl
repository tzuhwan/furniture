(define (problem table-assembly-full)
	(:domain furniture-assembly)
	(:objects
		left-plane right-plane top-plane - desk-plane
		drawer - desk-drawer
		left-plane-base right-plane-base top-plane-base drawer-base - handle-part
		left-plane-connect right-plane-connect drawer-connect - action-part
		top-plane-connector1 top-plane-connector2 top-plane-connector3 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor left-plane)
		(on-floor right-plane)
		(on-floor top-plane)
        (on-floor drawer)

		(clear left-plane)
		(clear right-plane)
		(clear top-plane)
        (clear drawer)

		(empty robot-gripper)

		(part-of left-plane-base left-plane)
		(part-of right-plane-base right-plane)
		(part-of top-plane-base top-plane)
        (part-of drawer-base drawer) 

		(part-of left-plane-connect left-plane)
		(part-of right-plane-connect right-plane)
		(part-of drawer-connect drawer)

        (part-of top-plane-connector1 top-plane)
        (part-of top-plane-connector2 top-plane)
        (part-of top-plane-connector3 top-plane)

		(affords-picking left-plane-base)
		(affords-picking right-plane-base)
		(affords-picking top-plane-base)
        (affords-picking drawer-base)

		(affords-connecting left-plane-connect)
        (affords-connecting right-plane-connect)
        (affords-connecting drawer-connect)
		(affords-connecting-to left-plane top-plane-connector1 top-plane)
        (affords-connecting-to right-plane top-plane-connector2 top-plane)
        (affords-connecting-to drawer top-plane-connector3 top-plane)
	)

	(:goal (and (connected-to left-plane top-plane)
				(connected-to right-plane top-plane)
                (connected-to drawer top-plane)
		   	)
	)
)