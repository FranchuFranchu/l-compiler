(chan () ("user"
    (pair
        (box . (either unit unit))
        (dual .  (box . (either unit unit)))
    ))
    (receive "user" "a"
        (duplicate "a" "b"
            (erase "a"
                (unbox "b"
                    (match "b"
                        (continue "b"
                            (cut (box () (right unit unit)) (var . "user"))
                        )
                        (continue "b"
                            (cut (box () (left unit unit)) (var . "user"))
                        )
                    )
                )
            )
        )
    )
)
