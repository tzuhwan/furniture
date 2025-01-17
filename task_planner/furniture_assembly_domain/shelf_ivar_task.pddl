(define (problem table-assembly-full)
	(:domain furniture-assembly)
	(:objects
		column - shelf-frame
		box1 box2 box3 box4 box5 - shelf-box
		column-base box1-base box2-base box3-base box4-base box5-base - handle-part
		box1-connect box2-connect box3-connect box4-connect box5-connect - action-part
		column-connector1 column-connector2 column-connector3 column-connector4 column-connector5 - tool-part
		robot-gripper - gripper
	)

	(:init
		(on-floor column)
		(on-floor box1)
		(on-floor box2)
		(on-floor box3)
		(on-floor box4)
		(on-floor box5)

		(clear column)
		(clear box1)
		(clear box2)
		(clear box3)
		(clear box4)
		(clear box5)

		(empty robot-gripper)

		(part-of column-base column)
		(part-of box1-base box1)
		(part-of box2-base box2)
		(part-of box3-base box3)
		(part-of box4-base box4)
		(part-of box5-base box5)
        (part-of box1-connect box1)
		(part-of box2-connect box2)
		(part-of box3-connect box3)
		(part-of box4-connect box4)
		(part-of box5-connect box5)

		(part-of column-connector1 column)
        (part-of column-connector2 column)
        (part-of column-connector3 column)
        (part-of column-connector4 column)
        (part-of column-connector5 column)

		(affords-picking column-base)
		(affords-picking box1-base)
		(affords-picking box2-base)
        (affords-picking box3-base)
		(affords-picking box4-base)
		(affords-picking box5-base)

		(affords-inserting box1-connect)
        (affords-inserting box2-connect)
        (affords-inserting box3-connect)
        (affords-inserting box4-connect)
        (affords-inserting box5-connect)
        
        (affords-inserting-into box1 column-connector1 column)
        (affords-inserting-into box2 column-connector2 column)
        (affords-inserting-into box3 column-connector3 column)
        (affords-inserting-into box4 column-connector4 column)
        (affords-inserting-into box5 column-connector5 column)
	)

	(:goal (and (inserted-into box1 column)
				(inserted-into box2 column)
                (inserted-into box3 column)
                (inserted-into box4 column)
                (inserted-into box5 column)
		   	)
	)
)