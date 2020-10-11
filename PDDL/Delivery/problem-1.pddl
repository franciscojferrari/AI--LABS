;; Problem definition
(define (problem problem-1)

  ;; Specifying the domain for the problem
  (:domain delivery-domain)

  ;; Objects definition
  (:objects
    ; DCs
    DC1
    DC2
    ; Areas
    A11
    A12
    A21
    A22
    ; Parcels
    parcel
    ; Vehicles
    LR
    SR1
    SR2
  )

  ;; Intial state of problem 1
  (:init
    ;; Declaration of the objects
    ; We initialize the DCs
    (DIST-CENTER DC1)
    (DIST-CENTER DC2)
    ; Areas
    (AREA A11)
    (AREA A12)
    (AREA A21)
    (AREA A22)
    ; Parcel
    (PARCEL parcel)
    ; Vehicles
    (VEHICLE LR)
    (VEHICLE SR1)
    (VEHICLE SR2)
    (long-range-vehicle LR)
    (short-range-vehicle SR1)
    (short-range-vehicle SR2)
    ; Roads
    (connected DC1 DC2) (connected DC2 DC1)
    (connected A11 A12) (connected A12 A11) 
    (connected DC1 A11) (connected A11 DC1)
    (connected DC1 A12) (connected A12 DC1)
    (connected A21 A22) (connected A22 A21) 
    (connected DC2 A21) (connected A21 DC2)
    (connected DC2 A22) (connected A22 DC2)
    
    ;; Declaration of the predicates of the objects
    ; We set vehicles locations
    (is-vehicle-at LR DC2)
    (is-vehicle-at SR1 DC1)
    (is-vehicle-at SR2 DC2)
    ; We set the parcel initial position
    (is-parcel-at parcel A11)
  )

  ;; Goal specification
  (:goal
    (and
      ; We want parcel delivered to A22
      (is-parcel-at parcel A22)
    )
  )

)
