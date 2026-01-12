package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
)

type GenerateRequest struct {
    Prompts          []string `json:"prompts"`
    Height           int      `json:"height,omitempty"`
    Width            int      `json:"width,omitempty"`
    GuidanceScale    float64  `json:"guidance_scale,omitempty"`
    NumInferenceSteps int     `json:"num_inference_steps,omitempty"`
    MaxSequenceLength int     `json:"max_sequence_length,omitempty"`
    Seed             int      `json:"seed,omitempty"`
}

func main() {
    url := "http://localhost:8000/generate"

    prompts := []string{
        "A vast rocky landscape dominated by towering, weathered stone formations, bathed in the ethereal glow of a vibrant night sky filled with a sea of stars, the Milky Way stretching across the heavens, captured from a low angle to emphasize the immense scale of the rocks against the expansive cosmos above. The scene is illuminated by soft, cool moonlight, casting long, dramatic shadows on the textured rock surfaces. The color palette is rich with deep blues, purples, and silvery whites, creating a serene, otherworldly atmosphere.",
        "A breathtaking scene of snow-capped mountains encircling a serene lake, their towering peaks perfectly mirrored in the still water under a twilight sky adorned with soft, colorful clouds. The gentle mist rises from the lake's surface, adding a mystical touch to the tranquil landscape, with the golden hues of the setting sun casting a warm glow over the scene. The composition captures the vastness of the mountains, the calm water, and the ethereal atmosphere, with a focus on soft, natural lighting and a cool color palette that enhances the peaceful mood.",
        "A serene snow-covered countryside unfolds in soft morning light, where a small group of woolly sheep graze in the crisp foreground, their breath visible in the cold air, while in the gently blurred midground, a charming, snow-dusted church with a stone steeple rises among rustic timber houses nestled among tall pine trees, all bathed in a pale blue winter palette under an overcast sky, captured with a 50mm lens for a shallow depth of field and a cinematic, filmic warmth.",
        "A confident woman with short, dark hair stands in a lush forest, her black dress flowing slightly in the breeze. She holds an owl with large, outstretched wings, its feathers sharp and detailed against the dappled sunlight. The light filters softly through the tall trees above, casting intricate shadows on her skin and highlighting her intricate tattoo along her arm. The forest around her feels serene and mystical, with the air thick with the scents of nature. The color palette blends deep greens, earthy browns, and the rich contrast of black and white.",
    }

    req := GenerateRequest{
        Prompts: prompts,
        Height:  4096,
        Width:   4096,
    }

    jsonData, err := json.Marshal(req)
    if err != nil {
        fmt.Println("Error marshaling JSON:", err)
        return
    }

    resp, err := http.Post(url, "application/json", bytes.NewBuffer(jsonData))
    if err != nil {
        fmt.Println("Error making request:", err)
        return
    }
    defer resp.Body.Close()

    _, err = io.ReadAll(resp.Body)
    if err != nil {
        fmt.Println("Error reading response body:", err)
        return
    }

    fmt.Println("Request sent, status:", resp.Status)
}