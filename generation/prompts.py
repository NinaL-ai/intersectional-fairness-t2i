class PromptBuilder:
    @staticmethod
    def baseline(profession, group):
        if group == "neutral":
            return (f"Photo portrait of a {profession}, "
                    f"centered, head-and-shoulders framing, looking at the camera, photorealistic")

        return (f"Photo portrait of a {group} {profession}, "
                f"centered, head-and-shoulders framing, looking at the camera, photorealistic")

    @staticmethod
    def sensitivity(profession, group):
        if group == "neutral":
            return [(f"Professional photograph of a {profession}, "
              f"centered, head-and-shoulders framing, looking at the camera, photorealistic"),
             (f"Portrait photograph of a {profession}, "
              f"centered, head-and-shoulders framing, looking at the camera, photorealistic")
             ]

        return [(f"Photo portrait of a {profession} who is {group}, "
                 f"centered, head-and-shoulders framing, looking at the camera, photorealistic"),
                (f"Photo portrait of a {profession}. The person is {group}. "
                 f"Centered, head-and-shoulders framing, looking at the camera, photorealistic")
                ]
