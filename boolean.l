(chan () ("user" (pair (either unit unit) (pair (either unit unit) (dual . (either unit unit))))) (receive "user" "a" (receive "user" "b"
    (match "a"
        (continue "a"
            (match "b"
                (continue "b"
                    (cut (var . "user") (left unit unit))
                )
                (continue "b"
                    (cut (var . "user") (left unit unit))
                )
            )
        )
        (continue "a"
           (cut (var . "user") (var . "b"))
        )
    )
)))
