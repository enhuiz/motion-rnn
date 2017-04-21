using UnityEngine;
using UnityEngine.UI;
using UnityEngine.EventSystems;
using System.Collections;
using System;

// control the joystick
public class Joystick : MonoBehaviour, IDragHandler, IPointerUpHandler, IPointerDownHandler {
    public static Joystick Instance { set; get; }
    private Image backgroundImage;
    private Image joystickImage;

    public Vector2 InputDirection { set; get; }

    void Awake() {
        Instance = this;
    }

    void Start() {
        backgroundImage = GetComponent<Image>();
        joystickImage = transform.GetChild(0).GetComponent<Image>();
        InputDirection = Vector3.zero;
    }

    public void SetDirection(Vector2 direction) {
        InputDirection = direction;
        InputDirection = (InputDirection.magnitude > 1) ? InputDirection.normalized : InputDirection;
        joystickImage.rectTransform.anchoredPosition = new Vector2(
            InputDirection.x * (backgroundImage.rectTransform.sizeDelta.x / 4),
            InputDirection.y * (backgroundImage.rectTransform.sizeDelta.y / 4));
    }

    public void OnDrag(PointerEventData eventData) {
        Vector2 pos = Vector2.zero;
        if (RectTransformUtility.ScreenPointToLocalPointInRectangle(
            backgroundImage.rectTransform,
            eventData.position,
            eventData.pressEventCamera,
            out pos)) {
            pos.x = (pos.x / backgroundImage.rectTransform.sizeDelta.x);
            pos.y = (pos.y / backgroundImage.rectTransform.sizeDelta.y);

            float x = (backgroundImage.rectTransform.pivot.x == 1) ? pos.x * 2 + 1 : pos.x * 2 - 1;
            float y = (backgroundImage.rectTransform.pivot.y == 1) ? pos.y * 2 + 1 : pos.y * 2 - 1;

            InputDirection = new Vector2(x, y);
            InputDirection = (InputDirection.magnitude > 1) ? InputDirection.normalized : InputDirection;
            joystickImage.rectTransform.anchoredPosition = new Vector2(
                InputDirection.x * (backgroundImage.rectTransform.sizeDelta.x /4),
                InputDirection.y * (backgroundImage.rectTransform.sizeDelta.y / 4));
        }
    }

    public void OnPointerDown(PointerEventData eventData) {
        OnDrag(eventData);
    }

    public void OnPointerUp(PointerEventData eventData) {
        InputDirection = new Vector2(0, 0);
        joystickImage.rectTransform.anchoredPosition = Vector2.zero;
    }
}
