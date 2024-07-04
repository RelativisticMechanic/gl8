#!/usr/bin/python3
# CHIP-8 Emulator Written in Python
# (C) Siddharth Gautam, 2024
# Usage: python gl8.py <PATH TO ROM>

# Hide PyGame splash
import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"

import pygame as pg
import numpy as np
from pygame.locals import DOUBLEBUF, OPENGL, HWSURFACE
import moderngl
import random
import sys
import io

def debug_print(s, end="\n"):
    print(s, end=end)
    

# CHIP-8 Font Data
CHIP8_SYSTEM_FONT = [
                        0xF0, 0x90, 0x90, 0x90, 0xF0, # 0
                        0x20, 0x60, 0x20, 0x20, 0x70, # 1
                        0xF0, 0x10, 0xF0, 0x80, 0xF0, # 2
                        0xF0, 0x10, 0xF0, 0x10, 0xF0, # 3 
                        0x90, 0x90, 0xF0, 0x10, 0x10, # 4
                        0xF0, 0x80, 0xF0, 0x10, 0xF0, # 5
                        0xF0, 0x80, 0xF0, 0x90, 0xF0, # 6
                        0xF0, 0x10, 0x20, 0x40, 0x40, # 7
                        0xF0, 0x90, 0xF0, 0x90, 0xF0, # 8
                        0xF0, 0x90, 0xE0, 0x90, 0xE0, # 9
                        0xF0, 0x90, 0xF0, 0x90, 0x90, # A
                        0xE0, 0x90, 0xE0, 0x90, 0xE0, # B
                        0xF0, 0x80, 0x80, 0x80, 0xF0, # C
                        0xE0, 0x90, 0x90, 0x90, 0xE0, # D
                        0xF0, 0x80, 0xF0, 0x80, 0xF0, # E
                        0xF0, 0x80, 0xF0, 0x80, 0x80, # F
                    ]

CHIP8_BEEP_SOUND = [82, 73, 70, 70, 246, 0, 0, 0, 87, 65, 86, 69, 102, 109, 116, 
                    32, 16, 0, 0, 0, 1, 0, 1, 0, 136, 21, 0, 0, 136, 21, 0, 0, 1, 
                    0, 8, 0, 100, 97, 116, 97, 210, 0, 0, 0, 168, 140, 105, 121, 
                    150, 126, 95, 99, 134, 141, 84, 89, 204, 154, 117, 102, 154, 
                    149, 136, 108, 78, 161, 102, 102, 147, 44, 96, 166, 120, 70, 
                    135, 89, 207, 129, 229, 72, 234, 175, 196, 91, 104, 123, 142, 
                    125, 107, 25, 49, 202, 227, 44, 41, 177, 208, 127, 103, 119, 
                    149, 167, 161, 153, 192, 213, 181, 160, 161, 171, 168, 150, 43, 
                    48, 53, 58, 64, 68, 72, 75, 78, 81, 84, 87, 89, 92, 94, 96, 98, 
                    100, 101, 103, 105, 106, 107, 109, 110, 111, 111, 111, 112, 113, 
                    114, 115, 116, 140, 208, 203, 198, 194, 190, 184, 171, 169, 166, 
                    164, 161, 176, 185, 181, 178, 175, 166, 129, 129, 129, 129, 129, 
                    133, 134, 134, 133, 133, 130, 122, 122, 123, 123, 123, 111, 100, 
                    102, 104, 105, 107, 115, 120, 121, 121, 122, 122, 131, 155, 153, 
                    151, 149, 148, 146, 139, 131, 131, 131, 131, 130, 130, 130, 86, 
                    89, 92, 95, 97, 99, 102, 104, 110, 114, 115, 116, 117, 118, 118, 
                    119, 120, 123, 155, 153, 151, 149, 147, 145, 143, 142, 141, 139, 
                    138, 137, 136, 135, 134, 133, 133, 132, 131, 131, 130, 130, 130, 
                    129, 129, 129, 128, 128, 128, 128]

CHIP8_MAX_MEM = 4096
CHIP8_MAX_STK = 16
CHIP8_SCREEN_WIDTH = 64
CHIP8_SCREEN_HEIGHT = 32
CHIP8_PIXEL_SIZE = 16
# Cycles per sec, must be greater than 60 as 60 Hz is minimum
CHIP8_CYCLES_PER_SEC = 600
CHIP8_BURN_IN_TIME = 2000

CHIP8_KEYMAP = {
    pg.K_1: 0,
    pg.K_2: 1,
    pg.K_3: 2,
    pg.K_4: 3,
    pg.K_q: 4,
    pg.K_w: 5,
    pg.K_e: 6,
    pg.K_r: 7,
    pg.K_a: 8,
    pg.K_s: 9,
    pg.K_d: 0xA,
    pg.K_f: 0xB,
    pg.K_z: 0xC,
    pg.K_x: 0xD,
    pg.K_c: 0xE,
    pg.K_v: 0xF
}

# Shader section begins here
# There are two sets of shaders. The first set is the basic shader,
# the second set is the post-processing shader that gives the CRT effect
vertex_shader_basic = """
#version 330
in vec2 in_vert;
in vec2 in_texcoord;
out vec2 v_text;

void main() {
    gl_Position = vec4(in_vert, 0.0, 1.0);
    v_text = in_texcoord;
}
"""

## Sets the color of the monitor sprites
fragment_shader_screen = """
#version 330
uniform sampler2D Texture;
in vec2 v_text;
out vec4 f_color;

void main() {
    float pixel = texture(Texture, vec2(v_text.x, 1 - v_text.y)).r;
    if (pixel > 0.0) {
        f_color = vec4(1.0, 0.80, 0.0, 1.0);  // Golden for active pixels
    } else {
        f_color = vec4(0.0, 0.0, 0.0, 0.0);  // Transparent for inactive pixels
    }
}
"""

fragment_shader_basic = """
#version 330
uniform sampler2D Texture;
in vec2 v_text;
out vec4 f_color;

void main() {
    f_color = texture(Texture, v_text);
}
"""

## The most important shader that makes the magic happen!
## Responsible for the CRT Effect
post_fragment_shader = """
#version 330
uniform sampler2D ScreenTexture;
uniform sampler2D PreviousFrame;
uniform float time;
uniform float current_frame_time;
uniform float previous_frame_time;

const float glow_size = 1.0 / 512.0;
const float intensity = 1.2;
const float flicker_fraction = 0.1;
const float flicker_speed = 20.0;
const float crt_noise_fraction = 0.20;
const float brightness_multiplier = 2.0;
const vec3 frame_color = vec3(0.5, 0.5, 0.51);
const vec3 back_color = vec3(0.4, 0.3, 0.0);
const vec3 noise_color = vec3(0.9, 0.8, 0.0);

float warp = 0.9;
float scan = 0.75;
float scanline_opacity = 0.1;
float scanline_density = 0.3;
float scanline_speed = 0.5;
float scanline_intensity = 0.15;
float scanline_spread = 0.2;
float vigenette_intensity = 0.25;
float vignette_brightness = 50.0;

// CRT Frame Settings
float frameShadowCoeff = 15.0;
float screenShadowCoeff = 15.0;
vec2 margin = vec2(0.03, 0.03);

in vec2 v_text;
out vec4 f_color;

vec4 burnInEffect(in vec2 uv, in sampler2D prev_frame, in sampler2D curr_frame)
{
    vec4 old_color = texture(prev_frame, uv);
    vec4 new_color = texture(curr_frame, uv);
    highp float decay = clamp((1.0/(current_frame_time - previous_frame_time)) * 120.0, 0.0, 1.0);
    old_color *= decay;
    return max(old_color, new_color);
}

/* The CRT glowing text effect, downsample, then upscale to cause a glowy blur */
vec4 crtGlow(in sampler2D crt_texture, in vec2 uv, in float blurSize)
{
    vec4 sum = vec4(0.0);
    sum += texture(crt_texture, vec2(uv.x - 4.0*blurSize, uv.y)) * 0.05;
    sum += texture(crt_texture, vec2(uv.x - 3.0*blurSize, uv.y)) * 0.09;
    sum += texture(crt_texture, vec2(uv.x - 2.0*blurSize, uv.y)) * 0.12;
    sum += texture(crt_texture, vec2(uv.x - blurSize, uv.y)) * 0.15;
    sum += texture(crt_texture, vec2(uv.x, uv.y)) * 0.16;
    sum += texture(crt_texture, vec2(uv.x + blurSize, uv.y)) * 0.15;
    sum += texture(crt_texture, vec2(uv.x + 2.0*blurSize, uv.y)) * 0.12;
    sum += texture(crt_texture, vec2(uv.x + 3.0*blurSize, uv.y)) * 0.09;
    sum += texture(crt_texture, vec2(uv.x + 4.0*blurSize, uv.y)) * 0.05;
        
    // blur in y (vertical)
    // take nine samples, with the distance blurSize between them
    sum += texture(crt_texture, vec2(uv.x, uv.y - 4.0*blurSize)) * 0.05;
    sum += texture(crt_texture, vec2(uv.x, uv.y - 3.0*blurSize)) * 0.09;
    sum += texture(crt_texture, vec2(uv.x, uv.y - 2.0*blurSize)) * 0.12;
    sum += texture(crt_texture, vec2(uv.x, uv.y - blurSize)) * 0.15;
    sum += texture(crt_texture, vec2(uv.x, uv.y)) * 0.16;
    sum += texture(crt_texture, vec2(uv.x, uv.y + blurSize)) * 0.15;
    sum += texture(crt_texture, vec2(uv.x, uv.y + 2.0*blurSize)) * 0.12;
    sum += texture(crt_texture, vec2(uv.x, uv.y + 3.0*blurSize)) * 0.09;
    sum += texture(crt_texture, vec2(uv.x, uv.y + 4.0*blurSize)) * 0.05;

    vec4 result = sum * (intensity + flicker_fraction * intensity * sin(time*flicker_speed)) + brightness_multiplier * burnInEffect(uv, ScreenTexture, PreviousFrame);
    return result;
}

float crtNoise(vec2 pos, float evolve) {
    
    // Loop the evolution (over a very long period of time).
    float e = fract((evolve*0.01));
    
    // Coordinates
    float cx  = pos.x*e;
    float cy  = pos.y*e;
    
    // Generate a "random" black or white value
    return fract(23.0*fract(2.0/fract(fract(cx*2.4/cy*23.0+pow(abs(cy/22.4),3.3))*fract(cx*evolve/pow(abs(cy),0.050)))));
}

vec3 vigenette(in vec2 uv, in vec3 oricol)
{
    float vig = (uv.x*uv.y - uv.x*uv.x*uv.y - uv.x*uv.y*uv.y + uv.x*uv.x*uv.y*uv.y) * vignette_brightness;
    vig = pow(vig, vigenette_intensity);
    return vig * oricol;
}

/* 
    The following code was borrowed from Cool-Retro-Term.
    It creates a nice frame around the terminal screen.
*/
float rand(vec2 co)
{
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

float max2(vec2 v)
{
    return max(v.x, v.y);
}

float min2(vec2 v)
{
    return min(v.x, v.y);
}

float prod2(vec2 v)
{
    return v.x * v.y;
}

float sum2(vec2 v)
{
    return v.x + v.y;
}

vec2 positiveLog(vec2 x) 
{
    return clamp(log(x), vec2(0.0), vec2(100.0));
}

vec4 crtFrame(in vec2 staticCoords, in vec2 uv)
{
    vec2 coords = uv * (vec2(1.0) + margin * 2.0) - margin;

    vec2 vignetteCoords = staticCoords * (1.0 - staticCoords.yx);
    float vignette = pow(prod2(vignetteCoords) * 15.0, 0.25);

    vec3 color = frame_color.rgb * vec3(1.0 - vignette);
    float alpha = 0.0;

    float frameShadow = max2(positiveLog(-coords * frameShadowCoeff + vec2(1.0)) + positiveLog(coords * frameShadowCoeff - (vec2(frameShadowCoeff) - vec2(1.0))));
    frameShadow = max(sqrt(frameShadow), 0.0);
    color *= frameShadow;
    alpha = sum2(1.0 - step(vec2(0.0), coords) + step(vec2(1.0), coords));
    alpha = clamp(alpha, 0.0, 1.0);
    alpha *= mix(1.0, 0.9, frameShadow);

    float screenShadow = 1.0 - prod2(positiveLog(coords * screenShadowCoeff + vec2(1.0)) * positiveLog(-coords * screenShadowCoeff + vec2(screenShadowCoeff + 1.0)));
    alpha = max(0.8 * screenShadow, alpha);

    vec4 final_color = vec4(color*alpha, alpha);
    return final_color;
}
/* End of cool-retro-term code */


void main() {
    // squared distance from center
    vec2 uv = v_text;
    vec2 dc = abs(0.5-uv);
    dc *= dc;
    
    // warp the fragment coordinates
    uv.x -= 0.5; uv.x *= 1.0+(dc.y*(0.3*warp)); uv.x += 0.5;
    uv.y -= 0.5; uv.y *= 1.0+(dc.x*(0.4*warp)); uv.y += 0.5;

    // sample inside boundaries, otherwise set to black
    if (uv.y > 1.0 || uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0)
        f_color = crtFrame(v_text, uv);
    else
    	{
            // determine if we are drawing in a scanline
            float apply = abs(sin(v_text.y)*0.5*scan);
            // sample the texture
            f_color = vec4(mix(crtGlow(ScreenTexture, uv, glow_size).rgb, vec3(0,0,0),apply),1.0);
            /* Add scanline */
            f_color.rgb += fract(smoothstep(-1.0, 0.0, uv.y - 1.0 * fract(time * 0.1976))) * scanline_intensity * back_color;
            /* Add noise */
            f_color.rgb = mix(f_color.rgb, noise_color * crtNoise(uv, time), crt_noise_fraction);
            f_color.rgb = vigenette(uv, f_color.rgb);
        }
}
"""

# CHIP-8 CPU Object
class Chip8Machine:
    def __init__(self):
        # RAM & General Purpose Regs
        self.memory = [0 for i in range(CHIP8_MAX_MEM)]
        self.v = [0 for i in range(16)]
        # IP (Instruction Pointer)
        self.ip = 0
        # Memory Access
        self.i = 0
        # Delay & Sound Timer
        self.dt = 0
        self.st = 0
        # Stack
        self.stack = [0 for i in range(CHIP8_MAX_STK)]
        self.sp = 0
        # Screen
        self.screen = np.array([0 for _ in range(CHIP8_SCREEN_HEIGHT*CHIP8_SCREEN_WIDTH)], dtype=np.uint8)
        # Keys
        self.keys = [0 for i in range(16)]
        # Waiting for keypress
        self.wait_until_input = False
        self.waiting_register = 0
        self.redraw = False

    def Initialize(self, rom_data):
        # Load Font Data
        for i in range(len(CHIP8_SYSTEM_FONT)):
            self.memory[i] = CHIP8_SYSTEM_FONT[i]
        
        # Load ROM data
        for i in range(len(rom_data)):
            self.memory[i+0x200] = rom_data[i]
        
        # Jump!
        self.ip = 0x200

    def IsWaiting(self):
        return self.wait_until_input
    
    def ClearScreen(self):
        self.screen = np.array([0 for _ in range(CHIP8_SCREEN_HEIGHT*CHIP8_SCREEN_WIDTH)], dtype=np.uint8)
    
    # Ensure all reigsters stick to their limits
    def MaintainSanity(self):
        for i in range(0, 16):
            self.v[i] = self.v[i] % 256
        self.i = self.i % 4096
        self.dt = self.dt % 256
        self.st = self.st % 256
        self.sp = self.sp % CHIP8_MAX_STK
        self.ip = self.ip % 4096

    def EmulateInstruction(self):
        self.MaintainSanity()

        debug_print(f'[{self.ip}] ', end="")
        byte_immediate = self.memory[self.ip]
        byte_next = self.memory[self.ip + 1]
        instruction = (byte_immediate << 8) | byte_next
        leading_nib = (instruction & 0xF000) >> 12

        self.ip += 2
        # One of CLS, SYS, or RET
        if(leading_nib == 0):
            match byte_next:
                case 0xE0:
                    # CLS
                    debug_print('CLS')
                    self.ClearScreen()
                case 0xEE:
                    # RET
                    debug_print('RET')
                    self.ip = self.stack[self.sp]
                    self.sp -= 1
                case _:
                    # SYS 0NNN
                    debug_print(f'SYS {instruction & 0x0FFF}')
        # JP NNN
        elif(leading_nib == 1):
            self.ip = instruction & 0x0FFF
            debug_print(f'JP {instruction & 0x0FFF}')
        # CALL NNN
        elif(leading_nib == 2):
            # Push the current IP to stack
            self.sp += 1
            self.stack[self.sp] = self.ip
            self.ip = instruction & 0x0FFF
            debug_print(f'CALL {instruction & 0x0FFF}')
        # SE Vx, KK
        elif(leading_nib == 3):
            reg = (instruction & 0x0F00) >> 8
            if(self.v[reg] == byte_next):
                self.ip += 2
            debug_print(f'SE V{reg}, {byte_next}')
        # SNE Vx, KK
        elif(leading_nib == 4):
            reg = (instruction & 0x0F00) >> 8
            if(self.v[reg] != byte_next):
                self.ip += 2
            debug_print(f'SNE V{reg}, {byte_next}')
        # SE Vx, Vy
        elif(leading_nib == 5):
            reg1 = (instruction & 0x0F00) >> 8
            reg2 = (instruction & 0x00F0) >> 4
            if(self.v[reg1] == self.v[reg2]):
                self.ip += 2
            debug_print(f'SE V{reg1}, V{reg2}')
        # LD Vx, KK
        elif(leading_nib == 6):
            reg = (instruction & 0x0F00) >> 8
            self.v[reg] = byte_next
            debug_print(f'LD V{reg}, {byte_next}')
        # ADD Vx, KK
        elif(leading_nib == 7):
            reg = (instruction & 0x0F00) >> 8
            self.v[reg] += byte_next
            debug_print(f'ADD V{reg}, {byte_next}')
        # 8XYN
        elif(leading_nib == 8):
            reg1 = (instruction & 0x0F00) >> 8
            reg2 = (instruction & 0x00F0) >> 4

            last_nib = instruction & 0x000F

            match last_nib:
                case 0:
                    self.v[reg1] = self.v[reg2]
                    debug_print(f'LD V{reg1}, V{reg2}')
                case 1:
                    self.v[reg1] |= self.v[reg2]
                    debug_print(f'OR V{reg1}, V{reg2}')
                case 2:
                    self.v[reg1] &= self.v[reg2]
                    debug_print(f'AND V{reg1}, V{reg2}')
                case 3:
                    self.v[reg1] ^= self.v[reg2]
                    debug_print(f'XOR V{reg1}, V{reg2}')
                case 4:
                    self.v[reg1] += self.v[reg2]
                    if(self.v[reg1] > 0xFF):
                        self.v[0xF] = 1
                    else:
                        self.v[0xF] = 0
                    debug_print(f'ADD V{reg1}, V{reg2}')
                case 5:
                    old_reg1 = self.v[reg1]
                    old_reg2 = self.v[reg2]
                    self.v[reg1] -= self.v[reg2]
                    if(old_reg1 > old_reg2):
                        self.v[0xF] = 1
                    else:
                        self.v[0xF] = 0
                    
                    debug_print(f'SUB V{reg1}, V{reg2}')
                case 6:
                    self.v[0xF] = self.v[reg1] & 0x1
                    self.v[reg1] = self.v[reg1] >> 1
                    debug_print(f'SHR V{reg1}')
                case 7:
                    old_reg1 = self.v[reg1]
                    old_reg2 = self.v[reg2]
                    self.v[reg1] = self.v[reg2] - self.v[reg1]

                    if(old_reg2 > old_reg1):
                        self.v[0xF] = 1
                    else:
                        self.v[0xF] = 0
                    
                    debug_print(f'SUBN V{reg1}, V{reg2}')
                case 0xE:
                    self.v[0xF] = self.v[reg1] & 0x1
                    self.v[reg1] = self.v[reg1] << 1
                    debug_print(f'SHL V{reg1}')
                case _:
                    debug_print(f'Unknown Instruction for 0x08 class')
        # SNE Vx, Vy
        elif(leading_nib == 9):
            reg1 = (instruction & 0x0F00) >> 8
            reg2 = (instruction & 0x00F0) >> 4
            if(self.v[reg1] != self.v[reg2]):
                self.ip += 2
        # LD I, NNN
        elif(leading_nib == 0xA):
            self.i = instruction & 0x0FFF
            debug_print(f'LD I, {instruction & 0x0FFF}')
        # JP V0, NNN
        elif(leading_nib == 0xB):
            self.ip = self.v[0] + (instruction & 0x0FFF)
            debug_print(f'JP V0, {instruction & 0x0FFF}')
        # RND Vx, KK
        elif(leading_nib == 0x0C):
            rnd = random.randint(0, 255)
            kk = instruction & 0x00FF
            reg = (instruction & 0x0F00) >> 8
            self.v[reg] = rnd & kk
            debug_print(f'RND V{reg}, {kk}')
        # DRW Vx, Vy, n
        elif(leading_nib == 0x0D):
            self.v[0xF] = 0
            regx = (instruction & 0x0F00) >> 8
            regy = (instruction & 0x00F0) >> 4
            n = (instruction & 0x000F)
            x, y = (self.v[regx], self.v[regy])
            x = x % CHIP8_SCREEN_WIDTH
            y = y % CHIP8_SCREEN_HEIGHT
            orig_x = x
            for i in range(0, n):
                pixel_row = self.memory[self.i + i]
                for j in range(0, 8):
                    if(pixel_row & (0x80 >> j)):
                        if(self.screen[y*CHIP8_SCREEN_WIDTH + x] != 0):
                            self.v[0xF] = 1
                    
                        self.screen[y*CHIP8_SCREEN_WIDTH + x] ^= 0xFF
                        
                    x += 1
                    if(x >= CHIP8_SCREEN_WIDTH):
                        x = 0
                y += 1
                x = orig_x
                if(y >= CHIP8_SCREEN_HEIGHT):
                    y = 0
            self.redraw = True
            debug_print(f'DRW V{regx}, V{regy}, {n}')
        # 0x0E class instructions
        elif(leading_nib == 0x0E):
            reg = (instruction & 0x0F00) >> 8
            match byte_next:
                # SKP Vx
                case 0x9E:
                    if(self.keys[self.v[reg] & 0x0F] == 1):
                        self.ip += 2
                    debug_print(f'SKP V{reg}')
                case 0xA1:
                    if(self.keys[self.v[reg] & 0x0F] == 0):
                        self.ip += 2
                    debug_print(f'SKNP V{reg}')
                case _:
                    debug_print(f'Unknown Instruction for 0x0E class')
        # 0x0F class instructions
        elif(leading_nib == 0x0F):
            reg = (instruction & 0x0F00) >> 8
            match byte_next:
                case 0x07:
                    self.v[reg] = self.dt
                    debug_print(f'LD V{reg}, DT')
                case 0x0A:
                    self.waiting_register = reg
                    self.wait_until_input = True
                    debug_print(f'WAITKEY V{reg}')
                case 0x15:
                    self.dt = self.v[reg]
                    debug_print(f'LD DT, V{reg}')
                case 0x18:
                    self.st = self.v[reg]
                    debug_print(f'LD ST, V{reg}')
                case 0x1E:
                    self.i = self.i + self.v[reg]
                    debug_print(f'ADD I, V{reg}')
                case 0x29:
                    self.i = 0x05 * (self.v[reg])
                    debug_print(f'LOADSPRITE V{reg}')
                case 0x33:
                    self.memory[self.i + 2] = self.v[reg] % 10
                    self.memory[self.i + 1] = (self.v[reg] // 10) % 10
                    self.memory[self.i] = (self.v[reg] // 100) % 10
                    debug_print(f'STBCD V{reg}')
                case 0x55:
                    for i in range(0, reg+1):
                        self.memory[self.i + i] = self.v[i]
                    debug_print(f'STREGS {reg}')
                case 0x65:
                    for i in range(0, reg+1):
                        self.v[i] = self.memory[self.i + i]
                    debug_print(f'LDREGS {reg}')
                case _:
                    debug_print(f'Unknown Instruction for 0x0F class')

def ReadChip8ROM(rom_file):
    rom_data = None
    try:
        with open(rom_file, 'rb') as f:
            rom_data = f.read()
    except:
        print(f'Error reading {rom_file}. Ensure the file exists.')
        exit(-1)
    return rom_data
            
def main(args):
    if len(args) != 2:
        print('Usage: python gl8.py <Chip 8 ROM>')
        return
    rom_data = ReadChip8ROM(args[1])
    c8 = Chip8Machine()
    c8.Initialize(rom_data)

    # Begin setting up PyGame and OpenGL
    pg.init()
    pg.mixer.init()
    pg.display.gl_set_attribute(pg.GL_MULTISAMPLEBUFFERS, 0)
    pg.display.gl_set_attribute(pg.GL_MULTISAMPLESAMPLES, 0)
    
    pygame_display = pg.display.set_mode((CHIP8_SCREEN_WIDTH * CHIP8_PIXEL_SIZE, CHIP8_SCREEN_HEIGHT * CHIP8_PIXEL_SIZE), 
                                         DOUBLEBUF | OPENGL | HWSURFACE | pg.NOFRAME)
    pygame_clock = pg.time.Clock()

    ctx = moderngl.create_context()
    ctx.enable(moderngl.BLEND)
    ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
    

    # Create the programs, one for rendering, one for post-processing
    screen_render_program = ctx.program(vertex_shader=vertex_shader_basic, fragment_shader=fragment_shader_screen)
    post_process_program = ctx.program(vertex_shader=vertex_shader_basic, fragment_shader=post_fragment_shader)
    burnin_frame_program = ctx.program(vertex_shader=vertex_shader_basic, fragment_shader=fragment_shader_basic)

    # Screen geometry, represented by two triangles
    screen_geometry = np.array([
        # x, y, tex_x, tex_y
        -1.0, -1.0, 0.0, 0.0,
        1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
        1.0,  1.0, 1.0, 1.0,
    ], dtype='f4')
    

    vbo = ctx.buffer(screen_geometry)
    vao = ctx.vertex_array(screen_render_program, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])
    
    # Create a framebuffer object for off-screen rendering
    fbo_texture = ctx.texture((CHIP8_SCREEN_WIDTH * CHIP8_PIXEL_SIZE, CHIP8_SCREEN_HEIGHT * CHIP8_PIXEL_SIZE), 4)
    fbo = ctx.framebuffer(color_attachments=[fbo_texture])
    post_vao = ctx.vertex_array(post_process_program, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])
    burnin_frame_vao = ctx.vertex_array(burnin_frame_program, [(vbo, '2f 2f', 'in_vert', 'in_texcoord')])
    
    # Create texture which our screen will be rendered to
    texture = ctx.texture((CHIP8_SCREEN_WIDTH, CHIP8_SCREEN_HEIGHT), 1, dtype='f1')
    texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    
    # Create a previous frame texture for burn in effect
    previous_frame_texture = ctx.texture((CHIP8_SCREEN_WIDTH * CHIP8_PIXEL_SIZE, 
                                          CHIP8_SCREEN_HEIGHT * CHIP8_PIXEL_SIZE), 4)
    previous_frame_fbo = ctx.framebuffer(color_attachments=[previous_frame_texture])
    previous_frame_texture.filter = (moderngl.NEAREST, moderngl.NEAREST)
    previous_frame_time = pg.time.get_ticks()

    post_process_program['ScreenTexture'] = 0
    post_process_program['PreviousFrame'] = 1

    # Set up audio
    beep_data = bytearray(CHIP8_BEEP_SOUND)
    beep_virtual_file = io.BytesIO(beep_data)
    beep_snd = pg.mixer.Sound(beep_virtual_file)

    # Main loop
    running = True
    while running:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                running = False

            # Input logic for the CHIP-8 System
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    running = False

                if ev.key in CHIP8_KEYMAP:
                    c8.keys[CHIP8_KEYMAP[ev.key]] = 1
                    if c8.IsWaiting():
                        c8.v[c8.waiting_register] = CHIP8_KEYMAP[ev.key]
                        c8.wait_until_input = False

            elif ev.type == pg.KEYUP:
                if ev.key in CHIP8_KEYMAP:
                    c8.keys[CHIP8_KEYMAP[ev.key]] = 0
            
        if(not c8.IsWaiting()):
            for i in range(int(CHIP8_CYCLES_PER_SEC/60)):
                c8.EmulateInstruction()
        
        # Update the delay and sound timers
        if(c8.dt > 0):
            c8.dt -= 1
        
        if(c8.st > 0):
            if not pg.mixer.get_busy():
                # If sound is not playing, play it!
                beep_snd.play()
            c8.st -= 1

        current_time = pg.time.get_ticks()

        # Clear the previous frame buffer if we are past 500ms of delay
        if(current_time - previous_frame_time >= CHIP8_BURN_IN_TIME):
            previous_frame_time = current_time
            previous_frame_fbo.use()
            ctx.clear(0.0, 0.0, 0.0, 0.0)
        
        # Write to previous frame (for burn-in effect)
        previous_frame_fbo.use()
        fbo_texture.use(location=0)
        burnin_frame_vao.render(moderngl.TRIANGLE_STRIP) 
        
        # Write the CHIP-8 screen
        texture.write(c8.screen.tobytes())
        # Render to internal FBO
        fbo.use()
        ctx.clear(0.0, 0.0, 0.0, 0.3)
        texture.use()
        vao.render(moderngl.TRIANGLE_STRIP)
            
        # Update the time variable in the post-process shader
        post_process_program['time'] = current_time / 1000.0
        post_process_program['current_frame_time'] = current_time
        post_process_program['previous_frame_time'] = previous_frame_time
        
        # Now render to screen with the post-processing shader
        ctx.screen.use()
        fbo_texture.use(location=0)
        previous_frame_texture.use(location=1)
        post_vao.render(moderngl.TRIANGLE_STRIP)

        pg.display.flip()
        pygame_clock.tick(60)
        

if __name__ == '__main__':
    main(sys.argv)