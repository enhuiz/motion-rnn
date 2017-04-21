using UnityEngine;
using System.Collections;
using UnityEngine.UI;

// control the lower right button 
public class ModeButton : MonoBehaviour {

	void Start() {
        GetComponentInChildren<Text>().text = "Testing\n Click to train";
        GetComponent<Button>().onClick.AddListener(() => {
            Player.Instance.Training = !Player.Instance.Training;
            string text = Player.Instance.Training ? "Training\n Click to test" : "Testing\n Click to train";
            GetComponentInChildren<Text>().text = text;
            Joystick.Instance.SetDirection(new Vector2(0, 0));
        });
    }
}
