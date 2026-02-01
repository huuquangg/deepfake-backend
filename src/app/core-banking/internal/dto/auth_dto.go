package dto

// LoginResponse - Response trả về sau khi login thành công
type LoginResponse struct {
    Token string   `json:"token"`
    User  *UserDTO `json:"user"`
}

// RegisterResponse - Response trả về sau khi register thành công
type RegisterResponse struct {
    Token string   `json:"token"`
    User  *UserDTO `json:"user"`
}