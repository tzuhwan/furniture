(define (domain furniture-assembly)
	(:requirements :strips :typing)
	(:types
		tool tool-part - object
		table-top table-leg chair-base chair-leg chair-seat shelf-frame shelf-box desk-plane desk-drawer - tool
		handle-part action-part - tool-part
		gripper
	)

	(:predicates
		(on ?t1 - tool ?t2 - tool)
		(on-floor ?t - tool)
		(clear ?t - tool)
		(empty ?g - gripper)
		(in-hand ?g - gripper ?t - tool)
		(screwed-into ?t - tool ?s - tool)
        (inserted-into ?t - tool ?s - tool)
        (connected-to ?t - tool ?s - tool)
        (inserted-into-base ?t - tool ?s - chair-base)
        (inserted-into-top ?t - tool ?s - tool)
		(part-of ?tp - tool-part ?t - tool)
		(affords-picking ?tp - handle-part)
		(affords-screwing ?tp - action-part)
        (affords-inserting ?tp - action-part)
		(affords-hammering ?tp - action-part)
		(affords-turning-on ?tp - action-part)
		(affords-turning-off ?tp - action-part)
		(affords-placing ?t - tool)
		(affords-screwing-into ?s - tool ?tp - tool-part ?t - tool)
        (affords-inserting-into ?s - tool ?tp - tool-part ?t - tool)
	)

	(:action pick-up-from-tool
		:parameters (?g - gripper ?tp - handle-part ?t - tool ?s - tool)
		:precondition (and (clear ?t) (empty ?g) (on ?t ?s) (part-of ?tp ?t) (affords-picking ?tp))
		:effect (and (in-hand ?g ?t) (clear ?s) (not (clear ?t)) (not (empty ?g)) (not (on ?t ?s)))
	)

	(:action pick-up-from-floor
		:parameters (?g - gripper ?tp - handle-part ?t - tool)
		:precondition (and (clear ?t) (empty ?g) (on-floor ?t) (part-of ?tp ?t) (affords-picking ?tp))
		:effect (and (in-hand ?g ?t) (not (clear ?t)) (not (empty ?g)) (not (on-floor ?t)))
	)

	(:action put-on-floor
		:parameters (?g - gripper ?t - tool)
		:precondition (and (in-hand ?g ?t) (affords-placing ?t))
		:effect (and (on-floor ?t) (clear ?t) (empty ?g) (not (in-hand ?g ?t)))
	)

	(:action screw-into
		:parameters (?g - gripper ?tp - action-part ?tl - tool ?th - tool-part ?tt - tool)
		:precondition (and (in-hand ?g ?tl) (part-of ?tp ?tl) (affords-screwing ?tp) (part-of ?th ?tt) (affords-screwing-into ?tl ?th ?tt))
		:effect (and (screwed-into ?tl ?tt) (clear ?tl) (empty ?g) (not (in-hand ?g ?tl)))
	)

    (:action insert-into
		:parameters (?g - gripper ?tp - action-part ?tl - tool ?th - tool-part ?tt - tool)
		:precondition (and (in-hand ?g ?tl) (part-of ?tp ?tl) (affords-inserting ?tp) (part-of ?th ?tt) (affords-inserting-into ?tl ?th ?tt))
		:effect (and (inserted-into ?tl ?tt) (clear ?tl) (empty ?g) (not (in-hand ?g ?tl)))
	)

	(:action insert-into-base
		:parameters (?g - gripper ?tp - action-part ?tl - tool ?th - tool-part ?tt - chair-base)
		:precondition (and (in-hand ?g ?tl) (part-of ?tp ?tl) (affords-inserting ?tp) (part-of ?th ?tt) (affords-inserting-into ?tl ?th ?tt))
		:effect (and (inserted-into ?tl ?tt) (inserted-into-base ?tl ?tt) (clear ?tl) (empty ?g) (not (in-hand ?g ?tl)))
	)

	(:action insert-into-top
		:parameters (?g - gripper ?tp - action-part ?tl - tool ?th - tool-part ?tt - tool ?tb - chair-base)
		:precondition (and (in-hand ?g ?tl) (part-of ?tp ?tl) (affords-inserting ?tp) (part-of ?th ?tt) (affords-inserting-into ?tl ?th ?tt) (inserted-into-base ?tt ?tb))
		:effect (and (inserted-into ?tl ?tt) (inserted-into-top ?tl ?tt) (clear ?tl) (empty ?g) (not (in-hand ?g ?tl)))
	)

	(:action connect-two-to-stool
		:parameters (?g1 - gripper ?tp1a - action-part ?tl1 - tool ?ts1 - tool-part ?ts - chair-seat ?g2 - gripper ?tp2 - action-part ?tl2 - tool ?tp1b - tool-part)
		:precondition (and (in-hand ?g1 ?tl1) (in-hand ?g2 ?tl2)
						   (part-of ?tp1a ?tl1) (part-of ?tp1b ?tl1) (part-of ?tp2 ?tl2)
						   (affords-inserting ?tp1a) (affords-inserting ?tp2)
						   (part-of ?ts1 ?ts)
						   (affords-inserting-into ?tl1 ?ts1 ?ts) (affords-inserting-into ?tl2 ?tp1b ?tl1)
					  )
		:effect (and (connected-to ?tl1 ?ts) (connected-to ?tl2 ?tl1) (clear ?tl1) (clear ?tl2) (empty ?g1) (empty ?g2) (not (in-hand ?g1 ?tl1)) (not (in-hand ?g2 ?tl2)))
	)

	(:action connect-two-to-seat
		:parameters (?g1 - gripper ?tp1 - action-part ?tl1 - tool ?g2 - gripper ?tp2 - action-part ?tl2 - tool ?ts1 - tool-part ?ts2 - tool-part ?ts - chair-seat)
		:precondition (and (in-hand ?g1 ?tl1) (in-hand ?g2 ?tl2)
						   (part-of ?tp1 ?tl1) (part-of ?tp2 ?tl2)
						   (affords-inserting ?tp1) (affords-inserting ?tp2)
						   (part-of ?ts1 ?ts) (part-of ?ts2 ?ts)
						   (affords-inserting-into ?tl1 ?ts1 ?ts) (affords-inserting-into ?tl2 ?ts2 ?ts)
					  )
		:effect (and (connected-to ?tl1 ?ts) (connected-to ?tl2 ?ts) (clear ?tl1) (clear ?tl2) (empty ?g1) (empty ?g2) (not (in-hand ?g1 ?tl1)) (not (in-hand ?g2 ?tl2)))
	)
)