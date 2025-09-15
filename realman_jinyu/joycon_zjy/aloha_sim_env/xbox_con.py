#!/usr/bin/env python3
# xbox_pygame_probe.py
import pygame

def main():
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print('未检测到任何手柄')
        return
    pad = pygame.joystick.Joystick(0)
    pad.init()
    print(f'手柄名称: {pad.get_name()}')
    print('现在开始按按键，终端会打印“按键名 → button 编号”……')
    print('按窗口关闭或 Ctrl-C 退出\n')

    # 简单的事件循环
    screen = pygame.display.set_mode((300, 100))   # 必须创建窗口才能收事件
    running = True
    while running:
        for evt in pygame.event.get():
            if evt.type == pygame.QUIT:
                running = False
            elif evt.type == pygame.JOYBUTTONDOWN:
                # pygame 里 button 就是编号，直接取
                print(f'button {evt.button} 被按下')
            # 如果想看轴/帽子，也可以一并打印
            elif evt.type == pygame.JOYHATMOTION:
                print(f'hat {evt.hat} → {evt.value}')
            elif evt.type == pygame.JOYAXISMOTION:
                # 去抖：只打印变化大的
                if abs(evt.value) > 0.5:
                    print(f'axis {evt.axis} → {evt.value:.2f}')
    pygame.quit()

if __name__ == '__main__':
    main()