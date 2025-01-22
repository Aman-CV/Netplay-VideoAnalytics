# BADMINTON ANALYSIS


| Jobs  | File | Double | Singles | Approach | Comments |
| --- | --- | --- | --- | --- | --- |
| Player Tracking  | [Player Tracker](https://github.com/notcamelcase01/BadmintonCoachAI/blob/99eb3e7e93660b6be792be941231aea293f13ad2/trackers/player_tracker.py)  | ✅ | ✅ | YOLOv11.track pretrained | Tracks player effectively throught the video |
| Shuttle Tracking  | [Shuttle Tracker](https://github.com/notcamelcase01/BadmintonCoachAI/blob/99eb3e7e93660b6be792be941231aea293f13ad2/trackers/shuttle_tracker.py)  | ✅ | ✅ | WASB pretrained | Tracks shuttle effectively , however interpolation is pending when shuttle is out of camera view |
| Hit Frame | [Hit frame detector](https://github.com/notcamelcase01/BadmintonCoachAI/blob/99eb3e7e93660b6be792be941231aea293f13ad2/hitframe_detector/hitframe_detector.py) | ✅| :x: | X3D video classification fined tuned | 82% efficiency , yet to train with complete data |
| Court Key Point Detection | [Court key point detector](https://github.com/notcamelcase01/BadmintonCoachAI/blob/99eb3e7e93660b6be792be941231aea293f13ad2/court_detector/court_keypoints_detector.py) | :x: | :x: | Currently adding key points manully | Yet to prepare data and train RestNet model for point detection |
| Rally Separation | Pending... | :x: | :x: | Pending... | **Not much data available to test out possible solution** |

---

## HIGHLIGHTS 

- Using the hit frame and player position, we can estimate the shuttle's speed. A lower value indicates less reaction time, contributing to a better rally score.
- The number of hit frames in a rally also contributes to a better score.

---

### Sample Outputs


|idx|shot_no|player who shot|his position x|his position y|time stamp (frame no)|transformed X     |transformed Y     |distance         |time(seconds)              |speed            |
|------|-------|---------------|--------------|--------------|---------------------|------------------|------------------|-----------------|------------------|-----------------|
|0     |0.0    |1.0            |570.0         |570.0         |88.0                 |2.375474832437524 |2.3831420063390025|0.0              |0.0               |0.0              |
|1     |1.0    |4.0            |649.0         |377.0         |116.0                |3.1652963339203577|9.596093396928884 |7.256065446729178|0.9333333333333333|7.774355835781262|
|2     |2.0    |1.0            |570.0         |570.0         |152.0                |2.375474832437524 |2.3831420063390025|7.256065446729178|1.2               |6.046721205607648|
|3     |3.0    |4.0            |649.0         |377.0         |178.0                |3.1652963339203577|9.596093396928884 |7.256065446729178|0.8666666666666667|8.372383207764436|
|4     |4.0    |1.0            |570.0         |570.0         |213.0                |2.375474832437524 |2.3831420063390025|7.256065446729178|1.1666666666666667|6.219484668625009|
|5     |5.0    |4.0            |649.0         |377.0         |238.0                |3.1652963339203577|9.596093396928884 |7.256065446729178|0.8333333333333334|8.70727853607501 |

---

**Hit frame**

<img src="https://github.com/user-attachments/assets/9f01123d-5974-4033-aede-62fd9d235616" width=400/>

**Non hitframe**

<img src="https://github.com/user-attachments/assets/21c3a3f2-0e11-4864-a4bc-9b751c1f01cb" width=400/>





