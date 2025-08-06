(chan () ("user" (pair (box . (either unit unit)) (dual .  (pair (box . (either unit unit)) (box . (either unit unit))))))
    (receive "user" "a"
        (duplicate "a" "b"
            (cut (var . "user") (pair (var . "a") (var . "b")))
        )
    )
)
