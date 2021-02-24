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
        (connected-to ?t - tool ?s - tool)
		(part-of ?tp - tool-part ?t - tool)
		(affords-picking ?tp - handle-part)
		(affords-screwing ?tp - action-part)
        (affords-connecting ?tp - action-part)
		(affords-hammering ?tp - action-part)
		(affords-turning-on ?tp - action-part)
		(affords-turning-off ?tp - action-part)
		(affords-placing ?t - tool)
		(affords-screwing-into ?s - tool ?tp - tool-part ?t - tool)
        (affords-connecting-to ?s - tool ?tp - tool-part ?t - tool)
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

    (:action connect-to
		:parameters (?g - gripper ?tp - action-part ?tl - tool ?th - tool-part ?tt - tool)
		:precondition (and (in-hand ?g ?tl) (part-of ?tp ?tl) (affords-connecting ?tp) (part-of ?th ?tt) (affords-connecting-to ?tl ?th ?tt))
		:effect (and (connected-to ?tl ?tt) (clear ?tl) (empty ?g) (not (in-hand ?g ?tl)))
	)
)