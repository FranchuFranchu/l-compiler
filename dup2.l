(chan () ("user"
    (pair
        (box . (either unit unit))
        (dual .  unit)
    ))
    (receive "user" "a"
        (duplicate "a" "b"
            (erase "a"
                (erase "b"
                    (cut (var . "user") unit)
                )
            )
        )
    )
)
